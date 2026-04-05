"""20 diverse domains for retention experiment campaign.

Each domain defines:
- prompt: instruction with a semantic constraint the model must retain
- positive_keywords: tokens that indicate constraint is being followed
- negative_keywords: tokens that indicate constraint violation
- bias_profile_name: which BiasDomainProfile to use (maps in qwen_generation_bias.py)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionDomain:
    name: str
    prompt: str
    positive_keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...]
    bias_profile_name: str
    max_new_tokens: int = 500


RETENTION_DOMAINS: tuple[RetentionDomain, ...] = (
    # ── 1. Vegan chef (existing, baseline) ──────────────────────────
    RetentionDomain(
        name="vegan_meal_plan",
        prompt=(
            "You are a vegan chef. Write a detailed weekly meal plan "
            "with recipes for each day."
        ),
        positive_keywords=(
            "vegan", "plant-based", "tofu", "lentil", "lentils",
            "chickpea", "chickpeas", "bean", "beans", "vegetable",
            "vegetables", "mushroom", "mushrooms", "dairy-free",
        ),
        negative_keywords=(
            "meat", "chicken", "beef", "pork", "bacon", "fish",
            "salmon", "tuna", "shrimp", "egg", "eggs", "cheese",
            "butter", "milk", "cream", "yogurt", "sausage", "ham",
        ),
        bias_profile_name="vegan",
    ),
    # ── 2. FastAPI (existing, baseline) ─────────────────────────────
    RetentionDomain(
        name="fastapi_service",
        prompt=(
            "Write a complete async FastAPI service with typed Pydantic models, "
            "dependency injection, async request handlers, validation, background "
            "jobs, and deployment notes. Do not rewrite the design as Django or "
            "synchronous class-based views."
        ),
        positive_keywords=(
            "fastapi", "async", "await", "pydantic", "depends",
            "dependency", "httpexception", "response_model",
            "background", "uvicorn",
        ),
        negative_keywords=(
            "django", "flask", "jinja", "template", "synchronous",
            "render", "make_response", "wsgi",
        ),
        bias_profile_name="code",
    ),
    # ── 3. Proof by contradiction (existing, baseline) ──────────────
    RetentionDomain(
        name="proof_by_contradiction",
        prompt=(
            "Prove that the square root of 2 is irrational using "
            "proof by contradiction. Show every step clearly."
        ),
        positive_keywords=(
            "assume", "suppose", "contradiction", "therefore",
            "irrational", "rational", "coprime", "integer", "square",
        ),
        negative_keywords=(
            "theory", "algebraic", "polynomial", "field", "complex",
        ),
        bias_profile_name="math",
    ),
    # ── 4. Gluten-free bakery ───────────────────────────────────────
    RetentionDomain(
        name="gluten_free_bakery",
        prompt=(
            "You are a pastry chef specializing in gluten-free baking. "
            "Write 7 dessert recipes using only gluten-free flours. "
            "Never use wheat flour, barley, or rye in any recipe."
        ),
        positive_keywords=(
            "gluten-free", "almond flour", "rice flour", "oat flour",
            "coconut flour", "tapioca", "buckwheat", "cassava",
            "cornstarch", "potato starch",
        ),
        negative_keywords=(
            "wheat flour", "all-purpose flour", "bread flour",
            "barley", "rye", "semolina", "spelt", "durum",
        ),
        bias_profile_name="dietary",
    ),
    # ── 5. Rust without unsafe ──────────────────────────────────────
    RetentionDomain(
        name="rust_safe_only",
        prompt=(
            "Write a concurrent web scraper in Rust using tokio and reqwest. "
            "The entire codebase must be safe Rust — do not use any unsafe "
            "blocks, raw pointers, or transmute."
        ),
        positive_keywords=(
            "tokio", "reqwest", "async", "await", "arc", "mutex",
            "result", "error", "spawn", "join",
        ),
        negative_keywords=(
            "unsafe", "raw pointer", "*const", "*mut", "transmute",
            "union", "asm!",
        ),
        bias_profile_name="code_strict",
    ),
    # ── 6. GDPR compliance ──────────────────────────────────────────
    RetentionDomain(
        name="gdpr_data_policy",
        prompt=(
            "Write a data retention policy for a SaaS company that is fully "
            "GDPR-compliant. All personal data must be anonymized or deleted "
            "after 90 days. Never suggest storing personal data indefinitely "
            "or sharing it with third parties without explicit consent."
        ),
        positive_keywords=(
            "gdpr", "anonymize", "anonymized", "delete", "deletion",
            "consent", "data subject", "right to erasure", "retention period",
            "90 days", "data protection", "lawful basis",
        ),
        negative_keywords=(
            "indefinitely", "permanent", "forever", "sell data",
            "share without consent", "no expiration", "unlimited retention",
        ),
        bias_profile_name="legal",
    ),
    # ── 7. Metric units only ────────────────────────────────────────
    RetentionDomain(
        name="metric_units_only",
        prompt=(
            "Write a comprehensive guide to home renovation measurements. "
            "Use only metric units (meters, centimeters, kilograms, liters). "
            "Do not use any imperial units such as feet, inches, pounds, "
            "or gallons anywhere in the text."
        ),
        positive_keywords=(
            "meter", "meters", "centimeter", "centimeters", "millimeter",
            "kilogram", "kilograms", "liter", "liters", "celsius",
            "cm", "mm", "kg",
        ),
        negative_keywords=(
            "feet", "foot", "inches", "inch", "pounds", "pound",
            "gallons", "gallon", "fahrenheit", "ounce", "ounces",
            "yard", "yards", "mile", "miles",
        ),
        bias_profile_name="constraint",
    ),
    # ── 8. Halal cuisine ────────────────────────────────────────────
    RetentionDomain(
        name="halal_cuisine",
        prompt=(
            "You are a halal chef. Write a 5-day meal plan for Ramadan "
            "iftar dinners. All ingredients must be halal-certified. "
            "Never include pork, alcohol, or non-halal meat."
        ),
        positive_keywords=(
            "halal", "lamb", "chicken", "rice", "lentil", "lentils",
            "dates", "hummus", "tahini", "olive oil", "cumin",
            "iftar", "ramadan",
        ),
        negative_keywords=(
            "pork", "bacon", "ham", "sausage", "wine", "beer",
            "alcohol", "rum", "vodka", "gelatin", "lard",
        ),
        bias_profile_name="dietary",
    ),
    # ── 9. PostgreSQL without ORM ───────────────────────────────────
    RetentionDomain(
        name="postgresql_raw_sql",
        prompt=(
            "Write a database layer for a blog application using raw "
            "PostgreSQL queries with the psycopg2 library. Do not use "
            "any ORM such as SQLAlchemy, Django ORM, or Peewee. "
            "Write all queries as raw SQL."
        ),
        positive_keywords=(
            "psycopg2", "select", "insert", "update", "delete",
            "create table", "execute", "cursor", "connection",
            "commit", "rollback", "sql",
        ),
        negative_keywords=(
            "sqlalchemy", "django", "orm", "peewee", "session.query",
            "model.objects", "declarative_base", "mapped_column",
        ),
        bias_profile_name="code_strict",
    ),
    # ── 10. Renewable energy only ───────────────────────────────────
    RetentionDomain(
        name="renewable_energy_plan",
        prompt=(
            "Write an energy transition plan for a small city. "
            "The plan must rely exclusively on renewable energy sources: "
            "solar, wind, hydro, and geothermal. Do not recommend "
            "fossil fuels, nuclear power, or natural gas."
        ),
        positive_keywords=(
            "solar", "wind", "hydro", "geothermal", "renewable",
            "photovoltaic", "turbine", "battery storage",
            "sustainable", "clean energy",
        ),
        negative_keywords=(
            "coal", "oil", "natural gas", "fossil", "nuclear",
            "uranium", "diesel", "gasoline", "petroleum", "fracking",
        ),
        bias_profile_name="constraint",
    ),
    # ── 11. TypeScript strict mode ──────────────────────────────────
    RetentionDomain(
        name="typescript_strict",
        prompt=(
            "Write a REST API client library in TypeScript with strict "
            "null checks enabled. Every function must have explicit return "
            "types. Never use 'any' type, type assertions, or "
            "'@ts-ignore' comments."
        ),
        positive_keywords=(
            "typescript", "interface", "type", "string", "number",
            "boolean", "promise", "async", "await", "readonly",
            "undefined", "null",
        ),
        negative_keywords=(
            "any", "as any", "ts-ignore", "ts-nocheck", "eval",
            "object", ": Object",
        ),
        bias_profile_name="code_strict",
    ),
    # ── 12. Functional programming (no mutation) ────────────────────
    RetentionDomain(
        name="functional_no_mutation",
        prompt=(
            "Write a data processing pipeline in Python using only pure "
            "functions and immutable data. Do not use any mutable state, "
            "class instances, global variables, or in-place mutations. "
            "Use map, filter, reduce, and comprehensions."
        ),
        positive_keywords=(
            "map", "filter", "reduce", "lambda", "tuple", "frozenset",
            "comprehension", "pure", "immutable", "functools",
        ),
        negative_keywords=(
            "class ", "self.", "global ", ".append(", ".extend(",
            ".sort(", ".pop(", "del ", ".update(",
        ),
        bias_profile_name="code_strict",
    ),
    # ── 13. Formal academic tone ────────────────────────────────────
    RetentionDomain(
        name="formal_academic_style",
        prompt=(
            "Write a literature review on the effects of social media "
            "on adolescent mental health. Use formal academic language "
            "throughout. Never use first person (I, we, my), contractions "
            "(don't, can't, it's), or informal expressions."
        ),
        positive_keywords=(
            "research", "study", "findings", "literature", "evidence",
            "analysis", "methodology", "hypothesis", "significant",
            "participants", "furthermore", "moreover",
        ),
        negative_keywords=(
            " i ", " we ", " my ", " our ", "don't", "can't",
            "won't", "it's", "they're", "we're", "isn't",
            "doesn't", "shouldn't",
        ),
        bias_profile_name="constraint",
    ),
    # ── 14. Zero-waste lifestyle ────────────────────────────────────
    RetentionDomain(
        name="zero_waste_guide",
        prompt=(
            "Write a beginner's guide to zero-waste living. Every "
            "recommendation must avoid single-use plastics and disposable "
            "products. Never suggest plastic bags, plastic wrap, "
            "disposable cups, or single-use containers."
        ),
        positive_keywords=(
            "reusable", "compost", "biodegradable", "glass", "metal",
            "bamboo", "cloth", "refill", "bulk", "sustainable",
            "zero waste", "recycle",
        ),
        negative_keywords=(
            "plastic bag", "plastic wrap", "disposable", "single-use",
            "styrofoam", "paper plate", "paper cup", "throw away",
        ),
        bias_profile_name="constraint",
    ),
    # ── 15. Kubernetes (no Docker Compose) ──────────────────────────
    RetentionDomain(
        name="kubernetes_native",
        prompt=(
            "Write a production deployment guide for a microservices "
            "application using Kubernetes. Use native Kubernetes resources: "
            "Deployments, Services, ConfigMaps, Secrets, and Ingress. "
            "Do not use Docker Compose, docker-compose.yml, or docker run."
        ),
        positive_keywords=(
            "kubernetes", "kubectl", "deployment", "service", "pod",
            "configmap", "secret", "ingress", "namespace", "helm",
            "replicas", "container",
        ),
        negative_keywords=(
            "docker-compose", "docker compose", "docker run",
            "docker-compose.yml", "compose.yml", "swarm",
        ),
        bias_profile_name="code",
    ),
    # ── 16. Organic farming ─────────────────────────────────────────
    RetentionDomain(
        name="organic_farming",
        prompt=(
            "Write a seasonal planting guide for a small organic farm. "
            "All methods must be certified organic. Never recommend "
            "synthetic pesticides, chemical fertilizers, GMO seeds, "
            "or herbicides."
        ),
        positive_keywords=(
            "organic", "compost", "mulch", "crop rotation",
            "companion planting", "natural", "manure", "cover crop",
            "beneficial insects", "heirloom",
        ),
        negative_keywords=(
            "pesticide", "herbicide", "synthetic", "chemical fertilizer",
            "gmo", "roundup", "glyphosate", "insecticide",
        ),
        bias_profile_name="constraint",
    ),
    # ── 17. Medical: drug-free pain management ──────────────────────
    RetentionDomain(
        name="drug_free_pain_management",
        prompt=(
            "Write a guide to managing chronic back pain without "
            "pharmaceutical drugs. Focus only on non-pharmacological "
            "approaches: physical therapy, exercise, stretching, "
            "acupuncture, and lifestyle changes. Never recommend "
            "opioids, NSAIDs, or any prescription medication."
        ),
        positive_keywords=(
            "physical therapy", "exercise", "stretching", "yoga",
            "massage", "acupuncture", "posture", "ergonomic",
            "mindfulness", "heat therapy", "ice",
        ),
        negative_keywords=(
            "opioid", "ibuprofen", "aspirin", "naproxen",
            "acetaminophen", "prescription", "painkiller", "morphine",
            "codeine", "tramadol", "nsaid",
        ),
        bias_profile_name="medical",
    ),
    # ── 18. Budget travel (no luxury) ───────────────────────────────
    RetentionDomain(
        name="budget_travel",
        prompt=(
            "Write a 2-week travel guide for Southeast Asia on a strict "
            "budget of $30 per day. Every recommendation must be "
            "budget-friendly. Never suggest luxury hotels, business class "
            "flights, fine dining restaurants, or expensive tours."
        ),
        positive_keywords=(
            "budget", "hostel", "street food", "local bus",
            "cheap", "affordable", "free", "backpack",
            "guesthouse", "market", "walking",
        ),
        negative_keywords=(
            "luxury", "five-star", "5-star", "business class",
            "first class", "resort", "fine dining", "michelin",
            "limousine", "premium",
        ),
        bias_profile_name="constraint",
    ),
    # ── 19. Python type-safe (no dynamic typing) ────────────────────
    RetentionDomain(
        name="python_typed_dataclasses",
        prompt=(
            "Write a configuration management system in Python using "
            "dataclasses and type hints everywhere. Every function must "
            "have complete type annotations. Do not use plain dicts, "
            "untyped variables, or **kwargs for configuration."
        ),
        positive_keywords=(
            "dataclass", "dataclasses", "int", "str", "float", "bool",
            "list[", "dict[", "optional", "field", "type", "-> ",
        ),
        negative_keywords=(
            "**kwargs", "untyped", "dict()", "= {}", ": dict,",
        ),
        bias_profile_name="code",
    ),
    # ── 20. Minimalist UI design ────────────────────────────────────
    RetentionDomain(
        name="minimalist_ui_design",
        prompt=(
            "Write a design system specification for a minimalist web "
            "application. Use only a monochrome palette with one accent "
            "color. Prioritize whitespace, clean typography, and simple "
            "layouts. Never use gradients, shadows, animations, "
            "carousels, or more than 2 fonts."
        ),
        positive_keywords=(
            "whitespace", "minimal", "clean", "typography", "simple",
            "monochrome", "accent", "grid", "sans-serif", "readable",
        ),
        negative_keywords=(
            "gradient", "shadow", "animation", "carousel", "parallax",
            "neon", "glitter", "skeuomorphic", "ornament",
        ),
        bias_profile_name="constraint",
    ),
)


def get_domain_by_name(name: str) -> RetentionDomain | None:
    for domain in RETENTION_DOMAINS:
        if domain.name == name:
            return domain
    return None
