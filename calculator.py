from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re

from scipy.optimize import linprog

item_name_power_net_average = "Desc_PowerNetAverage_C"
item_name_power_gross_average = "Desc_PowerGrossAverage_C"
item_name_production_shard = "Desc_WAT1_C"

alien_power_augmenter_base_power = 500.0
alien_power_augmenter_power_multiplier_unfueled = 0.1
alien_power_augmenter_power_multiplier_fueled = 0.3

resource_sink_amount = 1200.0

@dataclass
class ExtractionRecipeTemplate:
    class_name: str
    product: str
    build_power_consumer_class_name: str
    build_extractor_class_name: str
    amount: float
    """
    Extraction rate on this node, taking into account the purity,
    with 1.0 mExtractCycleTime, and 100 % clock speed.
    """
    power_consumption_multiplier: float
    """
    For miners and pumps, this is 1.0.
    For resource well extractors, the power comes
    from the resource well pressurizer, so all extractors share
    the same power. Assuming all fracking extractors are used,
    and all use the same power, this multiplier is 1/n,
    where n is the number of wells for a given pressurizer.
    """

extraction_recipe_templates = []
extraction_recipe_templates.append(
    ExtractionRecipeTemplate(
        class_name="Recipe_Water_WaterPump_C",
        product="Desc_Water_C",
        build_power_consumer_class_name="Build_WaterPump_C",
        build_extractor_class_name="Build_WaterPump_C",
        amount=120.0,
        power_consumption_multiplier=1.0
    )
)
for item in [
    "OreBauxite", "OreGold", "Coal", "OreCopper", "OreIron",
    "Stone", "RawQuartz", "SAM", "Sulfur", "OreUranium",
]:
    for purity, amount in [("Impure", 30.0), ("Normal", 60.0), ("Pure", 120.0)]:
        for miner in ["MinerMk1", "MinerMk2", "MinerMk3"]:
            extraction_recipe_templates.append(
                ExtractionRecipeTemplate(
                    class_name=f"Recipe_{item}_{purity}_{miner}_C",
                    product=f"Desc_{item}_C",
                    build_power_consumer_class_name=f"Build_{miner}_C",
                    build_extractor_class_name=f"Build_{miner}_C",
                    amount=amount,
                    power_consumption_multiplier=1.0
                )
            )
for purity, amount in [("Impure", 60.0), ("Normal", 120.0), ("Pure", 240.0)]:
    extraction_recipe_templates.append(
        ExtractionRecipeTemplate(
            class_name=f"Recipe_LiquidOil_{purity}_OilPump_C",
            product="Desc_LiquidOil_C",
            build_power_consumer_class_name="Build_OilPump_C",
            build_extractor_class_name="Build_OilPump_C",
            amount=amount,
            power_consumption_multiplier=1.0
        )
    )
for group_size in [6]:
    for purity, amount in [("Impure", 30.0), ("Normal", 60.0), ("Pure", 120.0)]:
        extraction_recipe_templates.append(
            ExtractionRecipeTemplate(
                class_name=f"Recipe_LiquidOil_{purity}_FrackingExtractor_{group_size}_C",
                product="Desc_LiquidOil_C",
                build_power_consumer_class_name="Build_FrackingSmasher_C",
                build_extractor_class_name="Build_FrackingExtractor_C",
                amount=amount,
                power_consumption_multiplier=1.0 / group_size
            )
        )
for group_size in [6, 7, 8, 10]:
    for purity, amount in [("Impure", 30.0), ("Normal", 60.0), ("Pure", 120.0)]:
        extraction_recipe_templates.append(
            ExtractionRecipeTemplate(
                class_name=f"Recipe_NitrogenGas_{purity}_FrackingExtractor_{group_size}_C",
                product="Desc_NitrogenGas_C",
                build_power_consumer_class_name="Build_FrackingSmasher_C",
                build_extractor_class_name="Build_FrackingExtractor_C",
                amount=amount,
                power_consumption_multiplier=1.0 / group_size
            )
        )

@dataclass
class GeneratorGeothermalRecipeTemplate:
    class_name: str
    power_production: float

generator_geothermal_recipe_templates = []
#for purity, power_production in [("Impure", 100.0), ("Normal", 200.0), ("Pure", 400.0)]:
#    generator_geothermal_recipe_templates.append(
#        GeneratorGeothermalRecipeTemplate(
#            f"Recipe_GeoThermal_{purity}_C",
#            power_production
#        )
#    )
# It is easier to handle only one Geothermal recipe, so we only keep the normal.
generator_geothermal_recipe_templates.append(
    GeneratorGeothermalRecipeTemplate(
        f"Recipe_GeoThermal_C",
        200.0
    )
)

@dataclass
class Resource:
    amount: float
    recipes_class_name: tuple[str]

def build_overrides(raw_build: dict) -> dict:
    # There is an error in the Docs json file,
    # see https://questions.satisfactorygame.com/post/675210bcc3824959d213972a.
    if raw_build["ClassName"] == "Build_SmelterMk1_C":
        new_raw_build = deepcopy(raw_build)
        new_raw_build["mProductionShardSlotSize"] = "1"
        new_raw_build["mProductionShardBoostMultiplier"] = "1.000000"
        return new_raw_build
    return raw_build

@dataclass
class Recipe:
    class_name: str
    ingredients: dict[str, float]
    product: dict[str, float]
    duration: float
    produced_in: str

    power_consumption: float
    """
    Base power consumption, at 1.0 Clock Speed,
    and without any production amplification (Somersloops).
    """
    power_consumption_exponent: float

    power_production: float | None
    """
    Base power production, at 1.0 Clock Speed.
    """

    production_boost: float
    """Clock Speed."""

    production_shard_slot_size: int
    """
    Maximum production amplification.
    Maximum number of Somersloops allowed to be used.
    """
    production_shard_boost_multiplier: float
    production_boost_power_consumption_exponent: float
    production_shards: int
    """
    Production amplification.
    Number of Somersloops used.
    """

    @property
    def items(self) -> dict[str, float]:
        ingredient_multiplier = -self.production_boost * 60.0 / self.duration

        full_production_shard_boost_multiplier = (
            (1.0 + self.production_shards * self.production_shard_boost_multiplier)
            if self.production_shard_slot_size != 0 else
            1.0
        )
        product_multiplier = (
            (self.production_boost * 60.0 / self.duration)
            * full_production_shard_boost_multiplier
        )

        base_items: dict[str, int | float] = {}
        for item, amount in self.ingredients.items():
            base_items[item] = base_items.get(item, 0.0) + ingredient_multiplier * amount
        for item, amount in self.product.items():
            base_items[item] = base_items.get(item, 0.0) + product_multiplier * amount
        if (
            item_name_power_net_average in base_items
            or item_name_power_gross_average in base_items
            or item_name_production_shard in base_items
        ):
            raise ValueError(f"Error: invalid element in base items for {self.class_name}.")

        power_consumption = (
            self.power_consumption
            * self.production_boost ** self.power_consumption_exponent
            * full_production_shard_boost_multiplier ** self.production_boost_power_consumption_exponent
        )
        power_production = (
            (self.power_production if (self.power_production is not None) else 0.0)
            * self.production_boost
        )
        power = power_production - power_consumption

        return (
            base_items
            | {
                item_name_power_net_average: power,
                item_name_power_gross_average: max(power, 0),
                item_name_production_shard: -float(self.production_shards),
            }
        )

@dataclass
class Build:
    power_consumption: float
    """
    Base power consumption, at 1.0 Clock Speed,
    and without any production amplification (Somersloops).
    """
    power_consumption_exponent: float
    production_boost_power_consumption_exponent: float
    can_change_potential: bool
    """Whether Clock Speed can be changed."""
    min_potential: float
    """Minimum Clock Speed."""
    base_production_boost: float
    """Default Clock Speed."""
    production_shard_slot_size: int
    """
    Maximum production amplification.
    Maximum number of Somersloops allowed to be used.
    """
    production_shard_boost_multiplier: float

    extract_cycle_time: float | None
    """
    For extractor, this is the extraction cycle time,
    for other buildings it's None.
    """

    def average_power_consumption(
        self,
        variable_power_consumption_constant: float,
        variable_power_consumption_factor: float,
    ) -> float:
        return self.power_consumption

@dataclass
class BuildHadronCollider(Build):
    def average_power_consumption(
        self,
        variable_power_consumption_constant: float,
        variable_power_consumption_factor: float,
    ) -> float:
        # For Build_HadronCollider_C (Particle Accelerator),
        # we assume the formula for power is
        # `constant + factor * t/duration`,
        # so the average power is 
        # `constant + factor/2`.
        return (
            variable_power_consumption_constant
            + variable_power_consumption_factor / 2
        )

@dataclass
class BuildConverter(Build):
    def average_power_consumption(
        self,
        variable_power_consumption_constant: float,
        variable_power_consumption_factor: float,
    ) -> float:
        # For Build_Converter_C (Converter),
        # we assume the formula for power is
        # `(constant + factor/2) + (factor/2) * sin(2*pi * t/duration)`,
        # so the average power is 
        # `constant + factor / 2`.
        return (
            variable_power_consumption_constant
            + variable_power_consumption_factor / 2
        )

@dataclass
class BuildQuantumEncoder(Build):
    def average_power_consumption(
        self,
        variable_power_consumption_constant: float,
        variable_power_consumption_factor: float,
    ) -> float:
        # For Build_QuantumEncoder_C (Quantum Encoder),
        # we assume the formula for power is a piecewise function
        # (https://satisfactory.wiki.gg/wiki/Quantum_Encoder#Power_fluctuation),
        # 0.0*duration < t < 0.1*duration, const + 0.5*fact,
        # 0.1*duration < t < 0.2*duration, const + 0.1*fact,
        # 0.2*duration < t < 0.3*duration, const + 0.4*fact,
        # 0.3*duration < t < 0.4*duration, const + 0.8*fact,
        # 0.4*duration < t < 0.5*duration, const + 0.6*fact,
        # 0.5*duration < t < 0.6*duration, const + 0.5*fact,
        # 0.6*duration < t < 0.7*duration, const + 0.9*fact,
        # 0.7*duration < t < 0.8*duration, const + 0.2*fact,
        # 0.8*duration < t < 0.9*duration, const + 1.0*fact,
        # 0.9*duration < t < 1.0*duration, const + 0.0*fact,
        # so the average power is 
        # `constant + factor / 2`.
        return (
            variable_power_consumption_constant
            + variable_power_consumption_factor / 2
        )

def _parse_ingredients_products(s: str, fluids_gases: set[str]) -> dict[str, float] | None:
    def find_in_parsed_tuple(
        key: str,
        parsed_tuple: list[tuple[str, str]]
    ) -> str | None:
        for first, second in parsed_tuple:
            if first == key:
                return second
        return None

    parsed_ingredients_products: dict[str, float] = {}
    tuples = re.findall(r'\(([^()]+)\)', s)
    for t in tuples:
        parsed_tuple = re.findall(r'([^=,]+)=([^=,]+)', t)
        item_class = find_in_parsed_tuple("ItemClass", parsed_tuple)
        if item_class is None:
            return None
        item_class = item_class.split(".")[-1][:-2]
        amount = float(find_in_parsed_tuple("Amount", parsed_tuple))
        amount /= 1000.0 if item_class in fluids_gases else 1.0
        parsed_ingredients_products[item_class] = amount
    return parsed_ingredients_products

def parse_produced_in(s: str) -> list[str]:
    allowed_produced_ins = []
    produced_ins = s[1:-1].split(",")
    for produced_in in produced_ins:
        produced_in = produced_in.split(".")[-1][:-1]
        if produced_in not in (
            "FGBuildGun",
            "BP_BuildGun_C",
            "BP_WorkBenchComponent_C",
            "BP_WorkshopComponent_C",
            "Build_AutomatedWorkBench_C",
            "FGBuildableAutomatedWorkBench",
        ):
            allowed_produced_ins.append(produced_in)
    return allowed_produced_ins

def parse_recipe(
    raw_recipe: dict,
    builds: dict[str, Build],
    fluids_gases: set[str],
    default_production_boost: float,
    production_amplified_production_boosts: list[float],
) -> list[Recipe]:
    class_name = raw_recipe["ClassName"]
    ingredients = _parse_ingredients_products(raw_recipe["mIngredients"], fluids_gases)
    if ingredients is None:
        raise ValueError(
            f"Error parsing {class_name} mIngredients."
        )
    product = _parse_ingredients_products(raw_recipe["mProduct"], fluids_gases)
    if product is None:
        raise ValueError(
            f"Error parsing {class_name} mProduct."
        )
    duration = float(raw_recipe["mManufactoringDuration"])
    produced_in = parse_produced_in(raw_recipe["mProducedIn"])
    if len(produced_in) > 1:
        raise ValueError(
            f"Error parsing {class_name} mProducedIn."
        )
    if len(produced_in) != 0 and produced_in[0] != "":
        build = builds[produced_in[0]]
        if default_production_boost < build.min_potential:
            raise ValueError(
                f"Error parsing {class_name}, {default_production_boost=} < {build.min_potential=}."
            )
        power_consumption = build.average_power_consumption(
            float(raw_recipe["mVariablePowerConsumptionConstant"]),
            float(raw_recipe["mVariablePowerConsumptionFactor"]),
        )
        new_recipes = []

        production_boost = (
            default_production_boost if build.can_change_potential
            else build.base_production_boost
        )
        new_recipes.append(
            Recipe(
                class_name=class_name,
                ingredients=ingredients,
                product=product,
                duration=duration,
                produced_in=produced_in[0],
                power_consumption=power_consumption,
                power_consumption_exponent=build.power_consumption_exponent,
                power_production=None,
                production_boost=production_boost,
                production_shard_slot_size=build.production_shard_slot_size,
                production_shard_boost_multiplier=build.production_shard_boost_multiplier,
                production_boost_power_consumption_exponent=build.production_boost_power_consumption_exponent,
                production_shards=0,
            )
        )

        for production_shards in range(1, build.production_shard_slot_size + 1):
            production_boosts = (
                production_amplified_production_boosts if build.can_change_potential
                else [build.base_production_boost]
            )
            for production_boost in production_boosts:
                new_recipes.append(
                    Recipe(
                        class_name=class_name + f"_PS{production_shards}_PB{production_boost}",
                        ingredients=ingredients,
                        product=product,
                        duration=duration,
                        produced_in=produced_in[0],
                        power_consumption=power_consumption,
                        power_consumption_exponent=build.power_consumption_exponent,
                        power_production=None,
                        production_boost=production_boost,
                        production_shard_slot_size=build.production_shard_slot_size,
                        production_shard_boost_multiplier=build.production_shard_boost_multiplier,
                        production_boost_power_consumption_exponent=build.production_boost_power_consumption_exponent,
                        production_shards=production_shards,
                    )
                )

        return new_recipes
    else:
        return []

def parse_build(class_name: str, raw_build: dict) -> Build:
    raw_build = build_overrides(raw_build)
    cls = {
        "Build_HadronCollider_C": BuildHadronCollider,
        "Build_Converter_C": BuildConverter,
        "Build_QuantumEncoder_C": BuildQuantumEncoder,
    }.get(class_name, Build)
    return cls(
        power_consumption=float(raw_build["mPowerConsumption"]),
        power_consumption_exponent=float(raw_build["mPowerConsumptionExponent"]),
        production_boost_power_consumption_exponent=float(raw_build["mProductionBoostPowerConsumptionExponent"]),
        min_potential=float(raw_build["mMinPotential"]),
        can_change_potential=(raw_build["mCanChangePotential"] == "True"),
        base_production_boost=float(raw_build["mBaseProductionBoost"]),
        production_shard_slot_size=int(raw_build["mProductionShardSlotSize"]),
        production_shard_boost_multiplier=float(raw_build["mProductionShardBoostMultiplier"]),
        extract_cycle_time=(float(raw_build["mExtractCycleTime"]) if "mExtractCycleTime" in raw_build else None),
    )

def parse_extraction_recipe(
    extraction_recipe_template: ExtractionRecipeTemplate,
    builds: dict[str, Build],
    default_production_boost: float,
    extractors_production_boost: float,
    production_amplified_production_boosts: list[float],
) -> list[Recipe]:
    new_recipes = []

    build_power_consumer = builds[extraction_recipe_template.build_power_consumer_class_name]
    build_extractor = builds[extraction_recipe_template.build_extractor_class_name]

    power_consumption = extraction_recipe_template.power_consumption_multiplier * build_power_consumer.power_consumption

    production_boost = build_power_consumer.base_production_boost
    if build_power_consumer.can_change_potential:
        production_boost = (
            extractors_production_boost
            if (extraction_recipe_template.build_extractor_class_name != "Build_WaterPump_C")
            else default_production_boost
        )

    class_name = extraction_recipe_template.class_name
    product = {
        extraction_recipe_template.product: extraction_recipe_template.amount / build_extractor.extract_cycle_time
    }
    duration = 60.0
    produced_in = extraction_recipe_template.build_extractor_class_name

    new_recipes.append(
        Recipe(
            class_name=class_name,
            ingredients={},
            product=product,
            duration=duration,
            produced_in=produced_in,
            power_consumption=power_consumption,
            power_consumption_exponent=build_power_consumer.power_consumption_exponent,
            power_production=None,
            production_boost=production_boost,
            production_shard_slot_size=build_power_consumer.production_shard_slot_size,
            production_shard_boost_multiplier=build_power_consumer.production_shard_boost_multiplier,
            production_boost_power_consumption_exponent=build_power_consumer.production_boost_power_consumption_exponent,
            production_shards=0,
        )
    )

    for production_shards in range(1, build_power_consumer.production_shard_slot_size + 1):
        production_boosts = (
            production_amplified_production_boosts if build_power_consumer.can_change_potential
            else [build_power_consumer.base_production_boost]
        )
        for production_boost in production_boosts:
            new_recipes.append(
                Recipe(
                    class_name=class_name + f"_PS{production_shards}_PB{production_boost}",
                    ingredients={},
                    product=product,
                    duration=duration,
                    produced_in=produced_in,
                    power_consumption=power_consumption,
                    power_consumption_exponent=build_power_consumer.power_consumption_exponent,
                    power_production=None,
                    production_boost=production_boost,
                    production_shard_slot_size=build_power_consumer.production_shard_slot_size,
                    production_shard_boost_multiplier=build_power_consumer.production_shard_boost_multiplier,
                    production_boost_power_consumption_exponent=build_power_consumer.production_boost_power_consumption_exponent,
                    production_shards=production_shards,
                )
            )

    return new_recipes

def parse_power_generator_recipe(
    raw_build: dict,
    default_production_boost: float,
    fluids_gases: set[str],
    energy_values: dict[str, float]
) -> list[Recipe]:
    new_recipes: list[Recipe] = []

    build_class_name = raw_build["ClassName"]
    power_production = float(raw_build["mPowerProduction"])

    if raw_build["mProductionShardSlotSize"] != "0":
        # We do not handle Production Shard (Somersloops)
        # in power generator buildings.
        raise ValueError(
            f"Error: mProductionShardSlot is invalid for {build_class_name}."
        )

    production_boost = (
        default_production_boost if (raw_build["mCanChangePotential"] == "True")
        else float(raw_build["mBaseProductionBoost"])
    )

    for raw_fuel in raw_build["mFuel"]:
        fuel_class = raw_fuel["mFuelClass"]

        energy_value = energy_values[fuel_class]
        if fuel_class in fluids_gases:
            energy_value *= 1000.0

        duration = energy_value / power_production

        ingredients = {fuel_class: 1.0}

        if raw_fuel["mSupplementalResourceClass"] != "":
            # This is only a guess that this is the general rule.
            # It works for all current generator types.
            # See also https://github.com/lunafoxfire/satisfactory-docs-parser/blob/v7.0.1/src/parsers/parseBuildables.ts#L310.
            supplemental_resource_amount_per_minute = (
                float(raw_build["mSupplementalToPowerRatio"]) * power_production * (3.0 / 50.0)
            )
            supplemental_resource_amount = duration * supplemental_resource_amount_per_minute / 60.0
            ingredients |= {raw_fuel["mSupplementalResourceClass"]: supplemental_resource_amount}

        product = {}
        if raw_fuel["mByproduct"] != "":
            product |= {raw_fuel["mByproduct"]: float(raw_fuel["mByproductAmount"])}

        new_recipes.append(
            Recipe(
                class_name=f"Recipe_{build_class_name}_{fuel_class}_C",
                ingredients=ingredients,
                product=product,
                duration=duration,
                produced_in=build_class_name,
                power_consumption=float(raw_build["mPowerConsumption"]),
                power_consumption_exponent=float(raw_build["mPowerConsumptionExponent"]),
                power_production=power_production,
                production_boost=production_boost,
                production_shard_slot_size=int(raw_build["mProductionShardSlotSize"]),
                production_shard_boost_multiplier=float(raw_build["mProductionShardBoostMultiplier"]),
                production_boost_power_consumption_exponent=float(raw_build["mProductionBoostPowerConsumptionExponent"]),
                production_shards=0,
            )
        )

    return new_recipes

def parse_geothermal_power_generator_recipe(
    raw_build: dict,
    default_production_boost: float
) -> list[Recipe]:
    new_recipes: list[Recipe] = []

    build_class_name = raw_build["ClassName"]

    if raw_build["mProductionShardSlotSize"] != "0":
        # We do not handle Production Shard (Somersloops)
        # in power generator buildings.
        raise ValueError(
            f"Error: mProductionShardSlot is invalid for {build_class_name}."
        )

    production_boost = (
        default_production_boost if (raw_build["mCanChangePotential"] == "True")
        else float(raw_build["mBaseProductionBoost"])
    )

    for generator_geothermal_recipe_template in generator_geothermal_recipe_templates:
        new_recipes.append(
            Recipe(
                class_name=generator_geothermal_recipe_template.class_name,
                ingredients={},
                product={},
                duration=60.0,
                produced_in=build_class_name,
                power_consumption=float(raw_build["mPowerConsumption"]),
                power_consumption_exponent=float(raw_build["mPowerConsumptionExponent"]),
                power_production=generator_geothermal_recipe_template.power_production,
                production_boost=production_boost,
                production_shard_slot_size=int(raw_build["mProductionShardSlotSize"]),
                production_shard_boost_multiplier=float(raw_build["mProductionShardBoostMultiplier"]),
                production_boost_power_consumption_exponent=float(raw_build["mProductionBoostPowerConsumptionExponent"]),
                production_shards=0,
            )
        )
    return new_recipes

def parse_resources_sink_recipes(
    raw_build: dict,
    sinkables: set[str]
) -> list[Recipe]:
    new_recipes: list[Recipe] = []

    build_class_name = raw_build["ClassName"]

    if raw_build["mProductionShardSlotSize"] != "0":
        # We do not handle Production Shard (Somersloops)
        # in power generator buildings.
        raise ValueError(
            f"Error: mProductionShardSlot is invalid for {build_class_name}."
        )

    if raw_build["mCanChangePotential"] == "True":
        # We do not handle changing clock resources
        # in Resource Sink.
        raise ValueError(
            f"Error: mCanChangePotential is invalid for {build_class_name}."
        )

    for item in sinkables:
        new_recipes.append(
            Recipe(
                class_name=f"Recipe_ResourceSink_{item}_C",
                ingredients={item: resource_sink_amount},
                product={},
                duration=60.0,
                produced_in=build_class_name,
                power_consumption=float(raw_build["mPowerConsumption"]),
                power_consumption_exponent=float(raw_build["mPowerConsumptionExponent"]),
                power_production=None,
                production_boost=float(raw_build["mBaseProductionBoost"]),
                production_shard_slot_size=int(raw_build["mProductionShardSlotSize"]),
                production_shard_boost_multiplier=float(raw_build["mProductionShardBoostMultiplier"]),
                production_boost_power_consumption_exponent=float(raw_build["mProductionBoostPowerConsumptionExponent"]),
                production_shards=0,
            )
        )

    return new_recipes

def parse_recipes(
    filename: str,
    default_production_boost: float,
    production_amplified_production_boosts: list[float],
    extractors_production_boost: float,
    default_generators_production_boost: float,
) -> list[Recipe]:
    recipes: list[Recipe] = []

    with open(filename, "r", encoding="utf-16") as f:

        docs = json.load(f)

        fluids_gases: set[str] = set()
        energy_values: dict[str, float] = {}
        sinkables: set[str] = set()
        for docs_item in docs:
            if docs_item["NativeClass"] in (
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGResourceDescriptor'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGItemDescriptor'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGItemDescriptorBiomass'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGItemDescriptorNuclearFuel'",
            ):
                for raw_item in docs_item["Classes"]:
                    class_name = raw_item["ClassName"]
                    if raw_item["mForm"] in ("RF_LIQUID", "RF_GAS"):
                        fluids_gases.add(class_name)
                    elif raw_item["mResourceSinkPoints"] != "0":
                        sinkables.add(class_name)
                    energy_values[class_name] = float(raw_item["mEnergyValue"])

        builds: dict[str, Build] = {}
        for docs_item in docs:
            if docs_item["NativeClass"] in (
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableManufacturer'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableManufacturerVariablePower'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableWaterPump'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableResourceExtractor'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableFrackingExtractor'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableFrackingActivator'",
            ):
                for raw_build in docs_item["Classes"]:
                    class_name = raw_build["ClassName"]
                    builds[class_name] = parse_build(class_name, raw_build)

        # Standard recipes.
        for docs_item in docs:
            if docs_item["NativeClass"] == "/Script/CoreUObject.Class'/Script/FactoryGame.FGRecipe'":
                for raw_recipe in docs_item["Classes"]:
                    new_recipes = parse_recipe(
                        raw_recipe,
                        builds,
                        fluids_gases,
                        default_production_boost,
                        production_amplified_production_boosts,
                    )
                    recipes.extend(new_recipes)

        # Extraction recipes.
        for extraction_recipe_template in extraction_recipe_templates:
            new_recipes = parse_extraction_recipe(
                extraction_recipe_template,
                builds,
                default_production_boost,
                extractors_production_boost,
                production_amplified_production_boosts,
            )
            recipes.extend(new_recipes)

        # Power generator recipes.
        for docs_item in docs:
            if docs_item["NativeClass"] in (
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableGeneratorFuel'",
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableGeneratorNuclear'",
            ):
                for raw_build in docs_item["Classes"]:
                    new_recipes = parse_power_generator_recipe(
                        raw_build,
                        default_generators_production_boost,
                        fluids_gases,
                        energy_values
                    )
                    recipes.extend(new_recipes)

        # Geothermal power generators recipes.
        for docs_item in docs:
            if docs_item["NativeClass"] in (
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableGeneratorGeoThermal'",
            ):
                for raw_build in docs_item["Classes"]:
                    if raw_build["ClassName"] == "Build_GeneratorGeoThermal_C":
                        new_recipes = parse_geothermal_power_generator_recipe(
                            raw_build,
                            default_generators_production_boost,
                        )
                        recipes.extend(new_recipes)

        # Resource sink recipes.
        for docs_item in docs:
            if docs_item["NativeClass"] in (
                "/Script/CoreUObject.Class'/Script/FactoryGame.FGBuildableResourceSink'",
            ):
                for raw_build in docs_item["Classes"]:
                    if raw_build["ClassName"] == "Build_ResourceSink_C":
                        new_recipes = parse_resources_sink_recipes(
                            raw_build,
                            sinkables,
                        )
                        recipes.extend(new_recipes)

    return recipes

def minimize_relative_item_availability(
    resources: dict[str, list[Resource] | float],
    recipes: list[Recipe],
) -> list[float]:
    """Minimize relative resource usage."""
    # It is a list of how much each recipe net produces the item.
    # scipy.linprog minimizes c*x.

    _check_valid_resources_recipes(resources, recipes)

    def get_recipe_index(class_name: str) -> int | None:
        for recipe_index, recipe in enumerate(recipes):
            if recipe.class_name == class_name:
                return recipe_index
        return None

    def get_maximum_resource_amount(item: str, resource: Resource) -> float:
        maximum_recipe_amount = None
        for recipe_class_name in resource.recipes_class_name:
            recipe_index = get_recipe_index(recipe_class_name)
            recipe_amount = recipes[recipe_index].items[item]
            if (maximum_recipe_amount is None) or (recipe_amount > maximum_recipe_amount):
                maximum_recipe_amount = recipe_amount
        return resource.amount * maximum_recipe_amount

    def get_maximum_item_amount(item: str, item_resources: list[Resource]) -> float | None:
        maximum_item_amount = 0.0
        for resource in item_resources:
            maximum_item_amount += get_maximum_resource_amount(item, resource)
        return maximum_item_amount

    c = [0.0] * len(recipes)
    for item, item_resources in resources.items():
        if isinstance(item_resources, float):
            if item_resources != math.inf:
                amount = item_resources
                for recipe_index, recipe in enumerate(recipes):
                    c[recipe_index] += recipe.items.get(item, 0.0) / amount
        else:
            maximum_item_amount = get_maximum_item_amount(item, item_resources)
            if maximum_item_amount != math.inf:
                for resource in item_resources:
                    for recipe_class_name in resource.recipes_class_name:
                        recipe_index = get_recipe_index(recipe_class_name)
                        c[recipe_index] += recipes[recipe_index].items.get(item, 0.0) / maximum_item_amount

    # Also minimalize the number of recipes used.
    epsilon = 1e-6
    for i in range(len(c)):
        c[i] += epsilon

    return c

def maximize_for(
    item: str,
    recipes: list[Recipe],
    resources: dict[str, list[Resource] | float | None],
) -> list[float]:
    """Calculate maximization coefficients."""
    # It is a list of how much each recipe net produces the item.
    # scipy.linprog minimizes c*x, so we negate some values here.

    _check_valid_resources_recipes(resources, recipes)

    c: list[int | float] = [-recipe.items.get(item, 0.0) for recipe in recipes]

    # A small epsilon is introduced, to secondarily minimize the relative source usage.
    epsilon = 1e-6
    c_minimize_relative_item_availability = minimize_relative_item_availability(recipes=recipes, resources=resources)
    for i in range(len(c)):
        c[i] += epsilon * c_minimize_relative_item_availability[i]

    return c

def maximize_for_augmented_power_net(
    alien_power_augmenters_unfueled: int,
    alien_power_augmenters_fueled: int,
    recipes: list[Recipe],
    resources: dict[str, list[Resource] | float | None],
) -> list[int | float]:
    """
    Calculate maximization coefficients if we want to maximize for augmented power.
    """
    # It is a list of how much each recipe net produces the item.
    # scipy.linprog minimizes c*x, so we negate some values here.
    # We maximize for
    # (G + 500 * (UAPA + FAPA)) * (1 + 0.1 * UAPA + 0.3 * FAPA) - (G - N),
    # where G is the gross power, N is the net power,
    # UAPA is the unfueled Alien Power Augmenters,
    # FAPA is the fueled Alien Power Augmenters.
    # This is the same as maximizing for
    # G * (0.1 * UAPA + 0.3 * FAPA) + N.
    c: list[int | float] = [
        (
            -recipe.items.get(item_name_power_gross_average, 0.0) * (
                (
                    alien_power_augmenter_power_multiplier_unfueled
                    * alien_power_augmenters_unfueled
                )
                + (
                    alien_power_augmenter_power_multiplier_fueled
                    * alien_power_augmenters_fueled
                )
            )
            -recipe.items.get(item_name_power_net_average, 0.0)
        )
        for recipe in recipes
    ]

    # A small epsilon is introduced, to secondarily minimize the relative source usage.
    epsilon = 1e-6
    c_minimize_relative_item_availability = minimize_relative_item_availability(recipes=recipes, resources=resources)
    for i in range(len(c)):
        c[i] += epsilon * c_minimize_relative_item_availability[i]

    return c

def _extract_data_from_solution(x: list[float], recipes: list[Recipe]) -> tuple[list[tuple[Recipe, float]], dict[str, float], dict[str, float]] | None:
    recipes_solution: list[tuple[Recipe, float]] = []
    items_solution: dict[str, float] = {}
    items_solution_gross: dict[str, float] = {}
    for recipe_index, recipe_amount in enumerate(x):
        if abs(recipe_amount) > 1e-10:
            recipe = recipes[recipe_index]
            recipes_solution.append((recipe, float(recipe_amount)))
            for item, item_amount in recipe.items.items():
                item_amount_total = recipe_amount * item_amount
                items_solution[item] = items_solution.get(item, 0.0) + item_amount_total
                if item_amount_total > 0:
                    items_solution_gross[item] = items_solution_gross.get(item, 0.0) + item_amount_total

    # Filtering out items that are there because of precision errors.
    items_solution_filtered = {}
    for item_key, item_value in items_solution.items():
       if abs(item_value) > 1e-10:
           items_solution_filtered[item_key] = float(item_value)
    items_solution_gross_filtered = {}
    for item_key, item_value in items_solution_gross.items():
       if abs(item_value) > 1e-10:
           items_solution_gross_filtered[item_key] = float(item_value)

    return recipes_solution, items_solution_filtered, items_solution_gross_filtered

def _check_valid_resources_recipes(
    resources: dict[str, list[Resource] | float | None],
    recipes: list[Recipe],
) -> None:
    # Check whether all recipes in the resources are valid.

    def get_recipe_index(class_name: str) -> int | None:
        for recipe_index, recipe in enumerate(recipes):
            if recipe.class_name == class_name:
                return recipe_index
        return None
    for item_resources in resources.values():
        if item_resources is not None and not isinstance(item_resources, float):
            for resource in item_resources:
                for recipe_class_name in resource.recipes_class_name:
                    if get_recipe_index(recipe_class_name) is None:
                        raise ValueError(
                            f"Error: invalid recipe name in resources {recipe_class_name}."
                        )

def solve(
    c: list[int | float],
    resources: dict[str, list[Resource] | float | None],
    items_equal: dict[str, int | float],
    recipes: list[Recipe],
    additional_recipes_equal: dict[str, float],
    zero_almost_all_items: bool,
) -> list[float] | None:
    """
    Calculates the optimal number of recipes, net items, gross items
    given the c as the number to optimize for (use for example maximize_for to get this).
    The other arguments are the recipes, and the requested minimum amount of items and item equality.
    If zero_almost_all_items is set, then all items, except for power and production shards will be constrained to zero.
    """

    _check_valid_resources_recipes(resources, recipes)

    # Calculate all items.
    items = []
    for recipe in recipes:
        for item in recipe.items.keys():
            if item not in items:
                items.append(item)


    # Calculate the upper bound matrix.
    # It is a matrix, a list of lists,
    # with each list corresponding to an item,
    # and tells how much each recipe produces an item type.
    # A_ub * x is a list for each item, how much each recipe produces,
    # and b_ub is the bound
    # A_ub * x <= b_ub.

    A_ub: list[list[int | float]] = []
    b_ub: list[int | float] = []

    unused_extraction_recipes_class_name: set[str] = {extraction_recipe_template.class_name for extraction_recipe_template in extraction_recipe_templates}
    # The number of extractors must be less than or equal to the number of nodes.
    for item, item_resources in resources.items():
        if not isinstance(item_resources, float):
            for resource in item_resources:
                if resource.amount != math.inf:
                    A_ub.append(
                        [
                            1.0 if (recipe.class_name in resource.recipes_class_name) else 0.0
                            for recipe in recipes
                        ]
                    )
                    b_ub.append(resource.amount)

                for recipe_class_name in resource.recipes_class_name:
                    unused_extraction_recipes_class_name.remove(recipe_class_name)
    # For the unused extraction recipes, we make sure there is less than or 0 of them.
    # (Practically they will be 0 then.)
    for extraction_recipe_class_name in unused_extraction_recipes_class_name:
        A_ub.append(
            [
                1.0 if (recipe.class_name == extraction_recipe_class_name) else 0.0
                for recipe in recipes
            ]
        )
        b_ub.append(0.0)
    # We restrict creating items out of nothing.
    # If there are no minimums for an item, then it must be bigger than 0.
    for item in items:
        if item not in items_equal:
            if isinstance(resources.get(item), float):
                if resources[item] != math.inf:
                    A_ub.append([-recipe.items.get(item, 0.0) for recipe in recipes])
                    b_ub.append(resources[item])
            else:
                A_ub.append([-recipe.items.get(item, 0.0) for recipe in recipes])
                b_ub.append(0.0)

    # Calculate the equality matrix.
    # It is a matrix, a list of lists,
    # with each list corresponding to an item,
    # and tells how much each recipe produces an item type.
    # A_eq * x is a list for each item, how much each recipe produces,
    # and b_eq is the bound
    # A_eq * x = b_eq.
    A_eq: list[list[int | float]] = []
    b_eq: list[int | float] = []
    for item in items:
        if (item_equal := items_equal.get(item)) is not None:
            A_eq.append([recipe.items.get(item, 0.0) for recipe in recipes])
            b_eq.append(item_equal)
    if zero_almost_all_items:
        for item in items:
            if item not in (item_name_power_net_average, item_name_power_gross_average, item_name_production_shard) and item not in items_equal:
                A_eq.append([recipe.items.get(item, 0.0) for recipe in recipes])
                b_eq.append(0.0)
    for additional_recipe_class_name, amount in additional_recipes_equal.items():
        A_eq.append(
            [
                1.0 if (recipe.class_name == additional_recipe_class_name) else 0.0
                for recipe in recipes
            ]
        )
        b_eq.append(amount)
    if len(A_eq) == 0:
        A_eq = b_eq = None

    solution = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )

    if not solution.success:
        print(solution)
        return None

    return _extract_data_from_solution(solution.x, recipes)

def get_power_augmented_net(
    items: dict[str, float],
    alien_power_augmenters_unfueled: int,
    alien_power_augmenters_fueled: int,
) -> float:
    power_gross = items[item_name_power_gross_average]
    power_net = items[item_name_power_net_average]
    power_consumed = power_gross - power_net

    alien_power_augmenter_multiplier = (
        alien_power_augmenter_power_multiplier_unfueled * alien_power_augmenters_unfueled
        + alien_power_augmenter_power_multiplier_fueled * alien_power_augmenters_fueled
    )

    power_augmented_gross = (
        (
            power_gross
            + alien_power_augmenter_base_power * (
                alien_power_augmenters_unfueled + alien_power_augmenters_fueled
            )
        )
        * (1 + alien_power_augmenter_multiplier)
    )
    power_augmented_net = power_augmented_gross - power_consumed
    return power_augmented_net

def get_power_augmented_gross(
    items: dict[str, float],
    alien_power_augmenters_unfueled: int,
    alien_power_augmenters_fueled: int,
) -> float:
    power_gross = items[item_name_power_gross_average]

    alien_power_augmenter_multiplier = (
        alien_power_augmenter_power_multiplier_unfueled * alien_power_augmenters_unfueled
        + alien_power_augmenter_power_multiplier_fueled * alien_power_augmenters_fueled
    )

    power_augmented_gross = (
        (
            power_gross
            + alien_power_augmenter_base_power * (
                alien_power_augmenters_unfueled + alien_power_augmenters_fueled
            )
        )
        * (1 + alien_power_augmenter_multiplier)
    )
    return power_augmented_gross
