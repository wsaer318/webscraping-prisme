# -*- coding: utf-8 -*-
"""
Générateur réaliste d'articles (tests 'chat ✕ lumière' vs hors-sujet) - VERSION AMÉLIORÉE

NOUVEAUTÉS v2.0:
- Articles de faible qualité (spam, gibberish, wrong_language, advertisement, clickbait)
- Articles multilingues (mélanges FR/EN + autres langues)
- HTML plus complexe (15+ types de tags, caractères spéciaux, symboles)
- Caractères Unicode exotiques (50+ caractères: âéîôû, αβγ, АБВ, 猫光, 🐱💡)
- URLs plus variées (6+ domaines, 8+ paramètres UTM, tracking)
- Plus d'auteurs (60 vs 12) et revues (50 vs 10)
- Plus de bruit (typos, HTML, unicode, longueurs extrêmes)
- Articles très courts (< 10 caractères) et très longs (3x normal)
- Langues étrangères (DE, ES, IT, PT, JA, ZH, RU, AR) pour tester langdetect
- Métadonnées enrichies (quality_type, embeddings support)

Fonctionnalités existantes:
- Multilingue FR/EN (+ quelques mélanges)
- Dates réalistes (2017–2025), auteurs, journaux, DOI optionnels
- Longueurs variées (abstract/body), HTML tags, emojis, caractères bizarres
- URLs propres + "sales" (utm, trailing /, www.), pour tester la normalisation
- Doublons exacts et quasi-doublons (titres/abstract/body légèrement modifiés)
- Quelques abstracts trop courts, langues non FR/EN et bruit lexical
- Sortie CSV avec: url, title, abstract, body, lang_hint, author, journal, published_at, doi, quality_type

Usage:
    python data.py --n-pos 200 --n-neg 150 --n-dupes 25 --n-near 40 --n-multilang 30 --n-low-quality 20 --seed 42 --out data/articles_fictifs.csv

Options:
    --n-pos 200           Articles pertinents (theme chat+lumiere)
    --n-neg 150           Articles hors-sujet
    --n-dupes 25          Doublons exacts du theme
    --n-near 40           Quasi-doublons du theme
    --n-multilang 30      Articles multilingues (melanges FR/EN)
    --n-low-quality 20    Articles de faible qualite (spam, gibberish, etc.)
"""

import argparse
import csv
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

# -------------------------- utilitaires texte --------------------------

def rand_date(start_year=2017, end_year=2025):
    d0 = datetime(start_year, 1, 1)
    d1 = datetime(end_year, 12, 31)
    delta = d1 - d0
    return d0 + timedelta(days=random.randint(0, delta.days))

def with_prob(p: float) -> bool:
    return random.random() < p

def inject_html_noise(text: str) -> str:
    # ajoute quelques tags bénins + espaces irréguliers
    decorations = [
        ("<i>", "</i>"), ("<b>", "</b>"), ("<em>", "</em>"), ("<u>", "</u>"),
        ("<span class='hl'>", "</span>")
    ]
    if not text:
        return text
    words = text.split()
    for _ in range(random.randint(0, min(3, max(0, len(words) // 10)))):
        i = random.randint(0, len(words) - 1)
        l, r = random.choice(decorations)
        words[i] = f"{l}{words[i]}{r}"
    s = " ".join(words)
    # espaces et ponctuation “bizarres”
    s = s.replace("'", "’")
    s = re.sub(r"\s{2,}", "  ", s)
    if with_prob(0.2):
        s += "  " + random.choice(["🙂", "📈", "🌙"])
    return s

def typo_perturb(s: str) -> str:
    # petites typos plausibles
    if not s or len(s) < 8:
        return s
    s = list(s)
    for _ in range(random.randint(1, 2)):
        i = random.randint(1, len(s) - 2)
        s[i], s[i+1] = s[i+1], s[i]
    return "".join(s)

def near_duplicate_text(s: str) -> str:
    # remplace quelques termes par synonymes proches
    repl = {
        "lumière": random.choice(["luminosité", "éclairage", "light"]),
        "chat": random.choice(["chat domestique", "félin", "cat"]),
        "nocturne": random.choice(["de nuit", "nocturnal", "sombre"]),
        "bleue": random.choice(["bleu", "blue"]),
        "cycle circadien": random.choice(["rythme circadien", "horloge biologique"]),
        "vision": random.choice(["perception visuelle", "acuité visuelle"]),
        "tapetum lucidum": random.choice(["tapetum", "couche réfléchissante"]),
    }
    out = s
    for k, v in repl.items():
        if with_prob(0.5):
            out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    if with_prob(0.4):
        out = inject_html_noise(out)
    if with_prob(0.3):
        out = typo_perturb(out)
    return out

def messy_url(base: str, i: int, themed: bool) -> str:
    # génère des variantes d'URL: www., trailing slash, utm...
    stem = base.rstrip("/")
    stem = random.choice([
        stem,
        stem.replace("https://", "https://www."),
        stem.replace("https://", "http://"),
        stem.replace("example.com", "example.org"),
        stem.replace("example.com", "testsite.net"),
    ])

    # Plus de paramètres UTM variés
    utm_sources = ["newsletter", "social", "email", "search", "direct", "referral"]
    utm_campaigns = ["testA", "campaignB", "promo2025", "research", "academic", "science"]
    utm_mediums = ["email", "cpc", "social", "organic", "referral"]

    q = []
    if with_prob(0.5):
        q.append(f"utm_source={random.choice(utm_sources)}")
    if with_prob(0.4):
        q.append(f"utm_medium={random.choice(utm_mediums)}")
    if with_prob(0.3):
        q.append(f"utm_campaign={random.choice(utm_campaigns)}")
    if with_prob(0.2) and themed:
        q.append("ref=cats")
    if with_prob(0.15):
        q.append(f"article_id={i}")
    if with_prob(0.1):
        q.append("tracking=true")

    query = ("?" + "&".join(q)) if q else ""
    trail = "/" if with_prob(0.5) else ""
    return f"{stem}_{i}{trail}{query}"

def inject_complex_html_noise(text: str) -> str:
    """Injecte du HTML plus complexe"""
    if not text:
        return text

    # HTML plus varié
    complex_decorations = [
        ("<i>", "</i>"), ("<b>", "</b>"), ("<em>", "</em>"), ("<u>", "</u>"),
        ("<strong>", "</strong>"), ("<span class='highlight'>", "</span>"),
        ("<sup>", "</sup>"), ("<sub>", "</sub>"), ("<mark>", "</mark>"),
        ("<small>", "</small>"), ("<big>", "</big>"), ("<tt>", "</tt>"),
        ("<code>", "</code>"), ("<kbd>", "</kbd>"), ("<var>", "</var>"),
    ]

    words = text.split()
    # Plus de modifications HTML
    for _ in range(random.randint(0, min(5, max(0, len(words) // 8)))):
        i = random.randint(0, len(words) - 1)
        l, r = random.choice(complex_decorations)
        words[i] = f"{l}{words[i]}{r}"

    s = " ".join(words)

    # Ajouter des caractères spéciaux et symboles
    special_chars = ["→", "←", "↑", "↓", "±", "≈", "≠", "≤", "≥", "×", "÷", "∞", "∑", "∏", "∆", "∇"]
    if with_prob(0.3):
        s = s.replace(".", random.choice([".", ".", ".", random.choice(special_chars) + "."]))

    # Espaces multiples et tabs
    s = re.sub(r"\s{2,}", lambda m: " " * random.randint(2, 5), s)
    if with_prob(0.2):
        s = s.replace(" ", "\t", random.randint(1, 3))

    return s

def inject_unicode_noise(text: str) -> str:
    """Injecte des caractères Unicode exotiques"""
    if not text or not with_prob(0.15):
        return text

    # Caractères unicode variés
    unicode_chars = [
        "â", "ê", "î", "ô", "û", "ä", "ë", "ï", "ö", "ü", "ÿ",
        "à", "è", "ì", "ò", "ù", "á", "é", "í", "ó", "ú",
        "ñ", "ç", "š", "ž", "ł", "ą", "ę", "ć", "ń", "ś", "ź",
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "μ", "ν", "ξ", "π", "ρ", "σ", "τ", "φ", "χ", "ψ", "ω",
        "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ы", "Э", "Ю", "Я",
        "猫", "狗", "光", "暗", "視", "覚", "行", "動", "生", "物",
        "🐱", "🐈", "💡", "🔆", "🌙", "⭐", "🌟", "✨", "🌞", "🌅"
    ]

    words = text.split()
    # Remplacer quelques mots par leur version unicode
    for _ in range(random.randint(1, min(3, len(words)))):
        i = random.randint(0, len(words) - 1)
        if len(words[i]) > 3 and with_prob(0.6):
            words[i] = words[i].replace(words[i][0], random.choice(unicode_chars))

    return " ".join(words)

def create_low_quality_article(lang: str) -> Dict[str, str]:
    """Génère un article de très faible qualité pour tester les filtres"""
    quality_types = [
        "spam", "gibberish", "too_short", "wrong_language", "corrupted",
        "advertisement", "clickbait", "unrelated_keywords"
    ]
    qtype = random.choice(quality_types)

    if qtype == "too_short":
        title = random.choice(["Chat", "Light", "Vision", "Study", "Research"])
        abstract = random.choice(["Short note.", "Brief study.", "Quick test."])
        body = "Very short content for testing filters."

    elif qtype == "spam":
        title = "Buy Now! Amazing Cat Light Vision Supplement - 50% Off!"
        abstract = "Revolutionary product! Transform your cat's vision overnight. Click here to order!"
        body = "This amazing supplement contains special ingredients that enhance feline night vision. Results guaranteed! Limited time offer. Call now!"

    elif qtype == "gibberish":
        title = "Xyz qwe rty uio pas dfg hjk lzxc"
        abstract = "Asd fgh jkl zxc vbn mqw erty uiop asdf ghjkl zxcvb nmqwe rtyui opasdf"
        body = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

    elif qtype == "wrong_language":
        # Langues non FR/EN
        other_langs = [
            ("de", "Die Katze und das Licht", "Diese Studie untersucht die Beziehung zwischen Katzen und Licht."),
            ("es", "El gato y la luz", "Este estudio examina cómo los gatos responden a la luz."),
            ("it", "Il gatto e la luce", "Questa ricerca analizza il comportamento dei gatti con la luce."),
            ("pt", "O gato e a luz", "Este estudo investiga como os gatos reagem à luz."),
            ("ja", "猫と光", "この研究は猫が光にどのように反応するかを調べます。"),
            ("zh", "猫和光", "这项研究调查猫如何对光做出反应。"),
            ("ru", "Кошка и свет", "Это исследование изучает, как кошки реагируют на свет."),
            ("ar", "القط والضوء", "تدرس هذه الدراسة كيفية تفاعل القطط مع الضوء.")
        ]
        lang_code, title, abstract = random.choice(other_langs)
        body = f"This is a {lang_code} article about cats and light. " * 20

    elif qtype == "advertisement":
        title = "Best Cat Food for Night Vision - Premium Formula!"
        abstract = "Discover the secret ingredient that makes cats see in the dark!"
        body = "Our premium cat food contains special nutrients that enhance nocturnal vision. Veterinarian approved! Order now and save 20%!"

    elif qtype == "clickbait":
        title = "You Won't Believe What Happens When Cats See This Light!"
        abstract = "Shocking discovery! Scientists reveal the truth about feline vision!"
        body = "In a groundbreaking study, researchers found that cats can actually see invisible light! This changes everything we know about animal vision!"

    else:  # unrelated_keywords
        title = random.choice(TITLES_NEG)
        abstract = random.choice(ABSTRACTS_NEG)
        body = random.choice(BODIES_NEG)

    return {
        "url": messy_url("https://spam-site.net/fake", random.randint(1000, 9999), themed=False),
        "title": title,
        "abstract": abstract,
        "body": body,
        "lang_hint": lang,
        "author": random.choice(AUTHORS_FR + AUTHORS_EN),
        "journal": random.choice(JOURNALS_FR + JOURNALS_EN),
        "published_at": rand_date().date().isoformat(),
        "doi": "",
        "quality_type": f"low_quality_{qtype}"
    }

def create_multilingual_article(i: int) -> Dict[str, str]:
    """Génère un article multilingue (mélange FR/EN)"""
    base_article = mk_positive(i)

    # Mélanger les langues dans le contenu
    mix_patterns = [
        "code_switch",  # Changements de langue brutaux
        "mixed_terms",  # Termes techniques en anglais
        "hybrid_title", # Titre mixte
        "bilingual_abstract"  # Abstract bilingue
    ]
    pattern = random.choice(mix_patterns)

    if pattern == "code_switch":
        # Changer des phrases entières de langue
        body_parts = base_article["body"].split(".")
        for j in range(len(body_parts)):
            if with_prob(0.4):
                if base_article["lang_hint"] == "fr":
                    body_parts[j] = body_parts[j].replace("chat", "cat").replace("lumière", "light").replace("vision", "vision")
                else:
                    body_parts[j] = body_parts[j].replace("cat", "chat").replace("light", "lumière").replace("vision", "vision")
        base_article["body"] = ". ".join(body_parts)

    elif pattern == "mixed_terms":
        # Garder la langue principale mais ajouter des termes techniques en anglais
        technical_terms = {
            "fr": ["photoreceptors", "tapetum lucidum", "circadian rhythm", "nocturnal vision", "scotopic vision"],
            "en": ["photorécepteurs", "tapetum lucidum", "rythme circadien", "vision nocturne", "vision scotopique"]
        }
        lang = base_article["lang_hint"]
        for term in technical_terms[lang]:
            if with_prob(0.3):
                base_article["body"] = re.sub(r'\b\w+\b', term, base_article["body"], count=1)

    elif pattern == "hybrid_title":
        # Titre avec des mots des deux langues
        if base_article["lang_hint"] == "fr":
            base_article["title"] = base_article["title"].replace("chat", "cat").replace("lumière", "light")
        else:
            base_article["title"] = base_article["title"].replace("cat", "chat").replace("light", "lumière")

    base_article["lang_hint"] = "mixed"
    base_article["quality_type"] = "multilingual"
    return base_article

# -------------------------- banques de contenu --------------------------

JOURNALS_FR = [
    "Revue de Comportement Animal", "Études Vétérinaires", "NeuroVision",
    "Biologie & Éclairage", "Chronobiologie Appliquée", "Journal Français de Neurosciences",
    "Revue Vétérinaire Moderne", "Optique et Vision Animale", "Éthologie Comparée",
    "Physiologie Comportementale", "Recherche en Ophtalmologie", "Animaux de Compagnie",
    "Sciences du Vivant", "Revue de Biologie", "Comportement et Adaptation",
    "Vision et Perception", "Chronobiologie Animale", "Recherche Vétérinaire",
    "Biologie Cellulaire", "Neurosciences Appliquées", "Écologie Comportementale",
    "Physiologie Animale", "Optométrie Vétérinaire", "Comportement des Mammifères",
    "Revue d'Éthologie", "Biologie de la Vision", "Adaptation Environnementale",
    "Sciences Comportementales", "Recherche en Physiologie", "Vision Nocturne"
]
JOURNALS_EN = [
    "Journal of Feline Studies", "Applied Chronobiology", "Vision & Perception",
    "Animal Behavior Letters", "Light & Biology", "Journal of Veterinary Science",
    "Modern Animal Behavior", "Optics and Animal Vision", "Comparative Ethology",
    "Behavioral Physiology", "Ophthalmology Research", "Companion Animal Journal",
    "Life Sciences Review", "Biology Journal", "Behavior and Adaptation",
    "Vision and Perception Quarterly", "Animal Chronobiology", "Veterinary Research",
    "Cellular Biology", "Applied Neurosciences", "Behavioral Ecology",
    "Animal Physiology", "Veterinary Optometry", "Mammalian Behavior",
    "Ethology Review", "Vision Biology", "Environmental Adaptation",
    "Behavioral Sciences", "Physiology Research", "Nocturnal Vision"
]

AUTHORS_FR = [
    "A. Martin", "L. Dupont", "S. Moreau", "C. Roux", "N. Lefèvre", "M. Rey",
    "P. Dubois", "J. Bernard", "M. Petit", "A. Durand", "C. Michel", "E. Girard",
    "F. André", "G. Thomas", "H. Simon", "I. Laurent", "J. Lefebvre", "K. Martin",
    "L. Dubois", "M. Moreau", "N. Roux", "O. Petit", "P. Durand", "Q. Michel",
    "R. Girard", "S. André", "T. Thomas", "U. Simon", "V. Laurent", "W. Lefebvre"
]
AUTHORS_EN = [
    "J. Smith", "K. Johnson", "E. Brown", "D. Wilson", "P. Clark", "R. Harris",
    "A. Davis", "B. Miller", "C. Wilson", "D. Moore", "E. Taylor", "F. Anderson",
    "G. Thomas", "H. Jackson", "I. White", "J. Harris", "K. Martin", "L. Thompson",
    "M. Garcia", "N. Martinez", "O. Robinson", "P. Clark", "Q. Rodriguez", "R. Lewis",
    "S. Lee", "T. Walker", "U. Hall", "V. Allen", "W. Young", "X. King"
]

TITLES_POS_FR = [
    "Les chats et la lumière naturelle",
    "Vision nocturne des félins",
    "Les photorécepteurs du chat",
    "Comportement des chats au soleil",
    "Influence de la lumière sur le sommeil du chat",
    "Adaptation visuelle du chat domestique aux environnements faiblement éclairés",
    "Effet de la lumière bleue sur l'activité nocturne féline",
    "Rythme circadien et exposition lumineuse chez Felis catus",
    "Tapetum lucidum et vision crépusculaire du chat",
    "Photopériode et comportement de chasse chez les félins domestiques",
    "Modulation de l'activité féline par l'intensité lumineuse",
    "Réponses comportementales aux variations de luminosité chez le chat",
    "Mécanismes de vision scotopique chez les félins",
    "Impact de l'éclairage artificiel sur le bien-être du chat",
    "Préférences lumineuses et zones de repos chez le chat domestique",
    "Sensibilité spectrale et perception colorée chez Felis catus",
    "Cycles d'activité féline en fonction de la photoperiode",
    "Architecture du sommeil félin sous différents régimes lumineux",
    "Photorecepteurs félins: rôles et distribution rétinienne",
    "Influence de la lumière lunaire sur le comportement nocturne des chats",
]
TITLES_POS_EN = [
    "Cats and Natural Light",
    "Feline Night Vision",
    "Photoreceptors in Domestic Cats",
    "Cat Behavior under Sunlight",
    "Light Exposure and Feline Sleep",
    "Visual Adaptation in Domestic Cats to Low-Light Environments",
    "Blue Light Effects on Nocturnal Feline Activity",
    "Circadian Rhythms and Light Exposure in Felis catus",
    "Tapetum Lucidum and Twilight Vision in Cats",
    "Photoperiod and Hunting Behavior in Domestic Felines",
    "Modulation of Feline Activity by Light Intensity",
    "Behavioral Responses to Luminance Variations in Cats",
    "Mechanisms of Scotopic Vision in Felines",
    "Impact of Artificial Lighting on Cat Welfare",
    "Light Preferences and Resting Areas in Domestic Cats",
    "Spectral Sensitivity and Color Perception in Felis catus",
    "Feline Activity Cycles as a Function of Photoperiod",
    "Sleep Architecture in Cats under Different Light Regimes",
    "Feline Photoreceptors: Roles and Retinal Distribution",
    "Lunar Light Influence on Nocturnal Cat Behavior",
]

ABSTRACTS_POS_FR = [
    "Cette étude explore la façon dont les chats réagissent aux variations de lumière dans leur environnement, en analysant leurs préférences comportementales et leurs adaptations physiologiques.",
    "Les félins possèdent une vision adaptée à la faible luminosité grâce à leurs bâtonnets et à leur tapetum lucidum. Nous documentons les mécanismes cellulaires sous-jacents à cette adaptation remarquable.",
    "La lumière influence le cycle circadien et l'activité des chats domestiques. Cette recherche quantifie l'impact de différentes intensités lumineuses sur les patterns d'activité diurnes et nocturnes.",
    "Nous analysons le rôle de la lumière sur la perception visuelle et comportementale du chat en utilisant des mesures comportementales et électrophysiologiques combinées.",
    "Une exposition prolongée à la lumière bleue modifie l'architecture du sommeil chez le chat. Les enregistrements polysomnographiques révèlent des perturbations significatives du sommeil paradoxal.",
    "L'adaptation des photorécepteurs félins aux environnements nocturnes représente un modèle évolutif fascinant. Nous caractérisons la distribution spatiale des cônes et bâtonnets dans la rétine centrale et périphérique.",
    "Le tapetum lucidum, structure réflective située derrière la rétine féline, augmente la sensibilité lumineuse d'un facteur six. Cette étude examine sa composition biochimique et son efficacité spectrale.",
    "Les cycles d'activité félins suivent un rythme bimodal avec des pics crépusculaires. Nous démontrons que l'intensité lumineuse ambiante module significativement ces patterns temporels.",
    "L'exposition à l'éclairage artificiel nocturne perturbe la sécrétion de mélatonine chez le chat domestique, avec des conséquences potentielles sur la régulation du sommeil et le métabolisme énergétique.",
    "Les chats présentent une sensibilité spectrale maximale autour de 500 nm, avec une vision dichromatique limitée. Nous explorons comment cette perception colorée influence le comportement de chasse.",
    "La photopériode saisonnière affecte le pelage, le comportement reproducteur et les niveaux d'activité chez les félins. Cette recherche longitudinale suit 50 chats sur 24 mois.",
    "Les mécanismes d'adaptation lumineuse chez le chat impliquent des ajustements pupillaires rapides et une modulation de la sensibilité rétinienne sur plusieurs échelles temporelles.",
    "Nous rapportons les premiers enregistrements électrorétinographiques haute résolution chez des chats exposés à différents spectres lumineux, révélant des réponses différentielles selon la longueur d'onde.",
    "L'influence de la lumière lunaire sur le comportement de chasse nocturne a été documentée par télémétrie GPS couplée à des capteurs d'accélération sur 30 chats semi-sauvages.",
    "Les zones de repos préférées par les chats domestiques sont significativement corrélées avec l'exposition solaire directe, suggérant une thermorégulation comportementale associée à la lumière.",
]
ABSTRACTS_POS_EN = [
    "This study explores how domestic cats respond to variations in ambient light, analyzing their behavioral preferences and physiological adaptations to different lighting conditions.",
    "Felines show enhanced low-light vision via rod-dense retinas and a reflective tapetum lucidum. We document the cellular mechanisms underlying this remarkable adaptation.",
    "Light exposure affects the circadian rhythms and daily activity of house cats. This research quantifies the impact of varying light intensities on diurnal and nocturnal activity patterns.",
    "We analyze how luminance shapes visual and behavioral responses in felines using combined behavioral and electrophysiological measurements.",
    "Prolonged blue-light exposure alters sleep architecture in cats. Polysomnographic recordings reveal significant disruptions in REM sleep patterns.",
    "The adaptation of feline photoreceptors to nocturnal environments represents a fascinating evolutionary model. We characterize the spatial distribution of rods and cones in central and peripheral retina.",
    "The tapetum lucidum, a reflective structure behind the feline retina, increases light sensitivity by a factor of six. This study examines its biochemical composition and spectral efficiency.",
    "Feline activity cycles follow a bimodal rhythm with crepuscular peaks. We demonstrate that ambient light intensity significantly modulates these temporal patterns.",
    "Exposure to artificial nocturnal lighting disrupts melatonin secretion in domestic cats, with potential consequences for sleep regulation and energy metabolism.",
    "Cats exhibit maximum spectral sensitivity around 500 nm, with limited dichromatic color vision. We explore how this color perception influences hunting behavior.",
    "Seasonal photoperiod affects coat, reproductive behavior, and activity levels in felines. This longitudinal study tracks 50 cats over 24 months.",
    "Light adaptation mechanisms in cats involve rapid pupillary adjustments and modulation of retinal sensitivity across multiple temporal scales.",
    "We report the first high-resolution electroretinographic recordings in cats exposed to different light spectra, revealing differential responses according to wavelength.",
    "The influence of lunar light on nocturnal hunting behavior was documented by GPS telemetry coupled with acceleration sensors on 30 semi-feral cats.",
    "Preferred resting areas in domestic cats are significantly correlated with direct sun exposure, suggesting behavioral thermoregulation associated with light.",
]

BODIES_POS_FR = [
    "Les yeux du chat, riches en bâtonnets, permettent une sensibilité accrue en faible lumière. "
    "Le tapetum lucidum réfléchit la lumière non absorbée. Les mesures actimétriques indiquent une "
    "augmentation de l'activité crépusculaire sous éclairage ambiant réduit. Introduction: La vision féline "
    "représente une adaptation remarquable aux environnements nocturnes. Matériel et méthodes: Nous avons suivi "
    "25 chats domestiques (Felis catus) pendant 12 semaines avec des colliers actimétriques et des caméras infrarouges. "
    "Résultats: La densité de bâtonnets atteint 450,000 cellules/mm² dans la zone centrale, soit 25 fois plus "
    "que chez l'humain. Le tapetum lucidum augmente la sensibilité d'un facteur 6 par réflexion des photons. "
    "Discussion: Ces adaptations expliquent pourquoi les chats sont particulièrement actifs au crépuscule et à l'aube.",

    "La phototransduction féline s'adapte rapidement aux transitions clair-obscur. Nous observons une "
    "modulation du rythme circadien sous éclairage LED, notamment en présence de spectres bleus (≈470 nm). "
    "Contexte: L'éclairage artificiel moderne peut perturber les rythmes biologiques naturels des félins. "
    "Méthodologie: Dix chats ont été exposés à trois conditions lumineuses: lumière naturelle, LED blanc chaud (3000K), "
    "et LED blanc froid (6500K). Des prélèvements sanguins horaires ont quantifié la mélatonine. "
    "Observations: L'exposition aux LED froides retarde la sécrétion de mélatonine de 90 minutes en moyenne. "
    "Les enregistrements comportementaux montrent un décalage significatif des pics d'activité (p<0.001). "
    "Implications: Les propriétaires devraient privilégier un éclairage chaud le soir pour respecter le cycle naturel.",

    "Des enregistrements polysomnographiques montrent une réduction du sommeil paradoxal après exposition "
    "à la lumière bleue le soir, avec une récupération partielle après 48 h d'obscurité contrôlée. "
    "Introduction: Le sommeil félin comprend plusieurs phases dont le sommeil paradoxal (REM) est crucial. "
    "Protocole expérimental: Huit chats adultes ont porté des électrodes EEG non invasives. Groupe contrôle: "
    "obscurité complète après 20h. Groupe test: exposition à lumière bleue (470nm, 100 lux) de 20h à 23h. "
    "Résultats quantitatifs: Le groupe test montre une réduction de 35% du temps en REM la première nuit (p=0.003), "
    "avec une latence d'endormissement augmentée de 25 minutes. L'analyse spectrale révèle une suppression des ondes "
    "thêta (4-8 Hz). Récupération: Après 48h d'obscurité, le sommeil REM revient à 85% du niveau basal. "
    "Conclusion: La lumière bleue perturbe significativement l'architecture du sommeil félin.",

    "La rétine féline contient une proportion exceptionnelle de photorécepteurs adaptés à la vision nocturne. "
    "Les bâtonnets représentent 96% des photorécepteurs, contre 95% chez l'humain, mais avec une densité "
    "absolue bien supérieure. Les cônes, bien que minoritaires, permettent une vision dichromatique avec deux types: "
    "cônes S (sensibles au bleu, pic à 450 nm) et cônes M (sensibles au vert, pic à 550 nm). L'absence de cônes L "
    "explique la perception limitée des rouges. Des expériences comportementales de discrimination colorée "
    "confirment que les chats distinguent le bleu du vert mais confondent rouge et vert. Cette configuration "
    "optimise la détection de mouvements en faible luminosité au détriment de la richesse chromatique.",

    "Le tapetum lucidum, structure multicouche située dans le fond de l'œil, fonctionne comme un miroir biologique. "
    "Composé de cellules contenant des cristaux de riboflavine et de zinc, il réfléchit sélectivement les longueurs "
    "d'onde entre 450 et 550 nm. Mesures spectrophotométriques: l'efficacité de réflexion atteint 90% dans le pic "
    "de sensibilité des bâtonnets. Cette adaptation double effectivement la probabilité qu'un photon soit capté. "
    "Effet secondaire: la diffusion lumineuse réduit légèrement l'acuité visuelle. Les chats voient donc moins "
    "net que les humains en plein jour (acuité: 20/100 vs 20/20), mais cette perte est négligeable dans leur "
    "niche écologique crépusculaire. Variations individuelles: la couleur du reflet (vert, jaune, orange) dépend "
    "de la composition exacte du tapetum et peut servir d'identification.",
]
BODIES_POS_EN = [
    "Cat retinas, with high rod density, increase sensitivity under dim light. The tapetum lucidum reflects "
    "unabsorbed photons. Actimetry shows increased twilight activity under reduced ambient illumination. "
    "Introduction: Feline vision represents a remarkable adaptation to nocturnal environments. Materials and methods: "
    "We tracked 25 domestic cats (Felis catus) for 12 weeks using actimetric collars and infrared cameras. "
    "Results: Rod density reaches 450,000 cells/mm² in the central area, 25 times higher than in humans. "
    "The tapetum lucidum increases sensitivity by a factor of 6 through photon reflection. Discussion: These "
    "adaptations explain why cats are particularly active at twilight and dawn.",

    "Feline phototransduction adapts swiftly to light–dark transitions. We observe circadian phase shifts "
    "under LED lighting, particularly with blue spectra (~470 nm). Context: Modern artificial lighting can "
    "disrupt natural biological rhythms in felines. Methodology: Ten cats were exposed to three lighting "
    "conditions: natural light, warm white LED (3000K), and cool white LED (6500K). Hourly blood samples "
    "quantified melatonin. Observations: Exposure to cool LEDs delays melatonin secretion by 90 minutes on average. "
    "Behavioral recordings show significant shifts in activity peaks (p<0.001). Implications: Owners should "
    "prefer warm lighting in the evening to respect the natural cycle.",

    "Polysomnography recordings indicate reduced REM sleep following evening blue-light exposure, with partial "
    "recovery after 48 h of controlled darkness. Introduction: Feline sleep includes several phases, of which "
    "REM sleep is crucial. Experimental protocol: Eight adult cats wore non-invasive EEG electrodes. Control "
    "group: complete darkness after 8 PM. Test group: exposure to blue light (470nm, 100 lux) from 8 PM to 11 PM. "
    "Quantitative results: The test group shows a 35% reduction in REM time the first night (p=0.003), with "
    "sleep latency increased by 25 minutes. Spectral analysis reveals theta wave suppression (4-8 Hz). Recovery: "
    "After 48h of darkness, REM sleep returns to 85% of baseline. Conclusion: Blue light significantly disrupts "
    "feline sleep architecture.",

    "The feline retina contains an exceptional proportion of photoreceptors adapted for night vision. Rods "
    "represent 96% of photoreceptors, compared to 95% in humans, but with much higher absolute density. Cones, "
    "although minority, enable dichromatic vision with two types: S cones (sensitive to blue, peak at 450 nm) "
    "and M cones (sensitive to green, peak at 550 nm). The absence of L cones explains limited red perception. "
    "Behavioral color discrimination experiments confirm that cats distinguish blue from green but confuse red "
    "and green. This configuration optimizes motion detection in low light at the expense of chromatic richness.",

    "The tapetum lucidum, a multilayer structure in the back of the eye, functions as a biological mirror. "
    "Composed of cells containing riboflavin and zinc crystals, it selectively reflects wavelengths between "
    "450 and 550 nm. Spectrophotometric measurements: reflection efficiency reaches 90% at the rod sensitivity "
    "peak. This adaptation effectively doubles the probability of photon capture. Side effect: light scattering "
    "slightly reduces visual acuity. Cats therefore see less sharply than humans in daylight (acuity: 20/100 vs "
    "20/20), but this loss is negligible in their crepuscular ecological niche. Individual variations: the reflection "
    "color (green, yellow, orange) depends on the exact tapetum composition and can serve as identification.",
]

# Hors-sujet / bruit (FR & EN)
TITLES_NEG = [
    "Les volcans d'Islande",
    "Recette du meilleur guacamole",
    "Les innovations en intelligence artificielle",
    "Tourisme durable en 2025",
    "Les effets du réchauffement climatique sur les oiseaux",
    "Architecture moderne à Tokyo",
    "L'histoire de la Renaissance italienne",
    "Techniques de méditation zen",
    "Blockchain et cryptomonnaies",
    "La physique quantique expliquée",
    "Jardinage urbain et permaculture",
    "Économie circulaire et recyclage",
    "Neurosciences et apprentissage",
    "Géopolitique du Moyen-Orient",
    "Cuisine moléculaire avancée",
    "Photographie de paysage",
    "Énergies renouvelables marines",
    "Sociologie des réseaux sociaux",
    "Archéologie précolombienne",
    "Psychologie cognitive",
    "L'impact du télétravail sur la productivité",
    "Les bienfaits du yoga pour la santé mentale",
    "Cryptomonnaies et avenir de la finance",
    "Intelligence artificielle en médecine",
    "Changement climatique et agriculture",
    "Réalité virtuelle dans l'éducation",
    "Microbiote intestinal et immunité",
    "Évolution des espèces selon Darwin",
    "Musique classique et développement cérébral",
    "Astrophysique des trous noirs",
    "Paléontologie et dinosaures",
    "Génétique et maladies héréditaires",
    "Océanographie et courants marins",
    "Botanique et plantes carnivores",
    "Météorologie et prévisions climatiques",
    "Chimie organique des polymères",
    "Géologie des plaques tectoniques",
    "Zoologie des mammifères marins",
    "Mathématiques fractales",
    "Physique des particules élémentaires",
    "Astronomie des exoplanètes",
    "Paléoclimatologie et ères glaciaires",
    "Biochimie des enzymes",
    "Géomorphologie des reliefs",
    "Écologie des écosystèmes forestiers",
    "Pharmacologie des antidépresseurs",
    "Histoire de l'art roman",
    "Anthropologie culturelle",
    "Sociolinguistique et dialectes",
    "Pédagogie Montessori",
    "Épistémologie des sciences"
]
ABSTRACTS_NEG = [
    "Cet article décrit les principaux volcans actifs d'Islande et leur impact géologique sur la région environnante, avec une analyse détaillée des éruptions récentes et de leurs conséquences environnementales.",
    "The traditional guacamole recipe with a modern twist, exploring various regional variations and ingredient substitutions, including nutritional analysis and cultural significance.",
    "AI models are transforming many industries through deep learning and natural language processing capabilities, with applications in healthcare, finance, and autonomous systems.",
    "Sustainable tourism is becoming a global priority as travelers seek eco-friendly alternatives to traditional vacation models, including carbon footprint reduction strategies.",
    "Migratory birds struggle to adapt to climate change, with shifting migration patterns observed across multiple species and potential impacts on biodiversity conservation.",
    "Modern architecture in Tokyo represents a fusion of traditional Japanese aesthetics and contemporary design principles, showcasing innovative building materials and urban planning.",
    "L'histoire de la Renaissance italienne révèle les innovations artistiques et scientifiques du 15ème siècle, avec un focus sur les contributions de Léonard de Vinci et Michel-Ange.",
    "Zen meditation techniques offer practical approaches to mindfulness and stress reduction in daily life, supported by clinical studies and neuroscientific research.",
    "Blockchain technology and cryptocurrencies are revolutionizing financial systems and digital transactions worldwide, with implications for banking, contracts, and digital identity.",
    "Quantum physics explained through accessible analogies and real-world applications of quantum mechanics, including quantum computing and cryptography.",
    "Urban gardening and permaculture methods enable sustainable food production in limited spaces, with techniques for soil improvement and water conservation.",
    "L'économie circulaire transforme les déchets en ressources à travers des systèmes de recyclage innovants, réduisant l'empreinte environnementale des industries.",
    "Neuroscience research reveals how brain plasticity influences learning and memory formation across lifespan, with applications in education and rehabilitation.",
    "Middle Eastern geopolitics involves complex interactions between historical conflicts and modern power dynamics, including oil politics and regional alliances.",
    "Molecular gastronomy applies scientific principles to create innovative culinary experiences and textures, using chemistry and physics in food preparation.",
    "Landscape photography techniques for capturing dramatic natural scenery in various lighting conditions, including composition rules and post-processing methods.",
    "Marine renewable energies harness ocean waves, tides, and thermal gradients for power generation, with environmental impact assessments and technological challenges.",
    "The sociology of social networks examines how digital platforms reshape human interactions and community formation in the digital age.",
    "Pre-Columbian archaeology uncovers advanced civilizations that flourished in the Americas before European contact, revealing sophisticated urban planning and agriculture.",
    "Cognitive psychology investigates mental processes including perception, attention, memory, and decision-making, with applications in human-computer interaction.",
    "L'impact du télétravail sur la productivité des employés a été étudié dans plusieurs entreprises, révélant des changements dans les patterns de communication et la satisfaction au travail.",
    "Les bienfaits du yoga pour la santé mentale incluent la réduction du stress et l'amélioration de la concentration, selon des études cliniques randomisées.",
    "Cryptomonnaies et avenir de la finance traditionnelle : analyse des technologies blockchain et de leur impact sur les systèmes bancaires centralisés.",
    "Intelligence artificielle en médecine : applications du machine learning dans le diagnostic médical et la découverte de médicaments.",
    "Changement climatique et agriculture : adaptation des pratiques culturales et développement de variétés résistantes aux nouvelles conditions environnementales.",
    "Réalité virtuelle dans l'éducation : immersion pédagogique et apprentissage expérientiel dans les sciences et l'histoire.",
    "Microbiote intestinal et immunité : rôle des bactéries commensales dans le développement et la maturation du système immunitaire.",
    "Évolution des espèces selon Darwin : mécanismes de sélection naturelle et spéciation observés dans les populations contemporaines.",
    "Musique classique et développement cérébral : effets des stimulations auditives sur la plasticité neuronale chez l'enfant.",
    "Astrophysique des trous noirs : formation, propriétés physiques et rôle dans l'évolution des galaxies.",
    "Paléontologie et dinosaures : nouvelles découvertes sur l'extinction du Crétacé et l'évolution des oiseaux modernes.",
    "Génétique et maladies héréditaires : technologies de séquençage et thérapies géniques pour les maladies monogéniques.",
    "Océanographie et courants marins : influence des gyres océaniques sur le climat global et la distribution des espèces marines.",
    "Botanique et plantes carnivores : adaptations morphologiques et physiologiques des Drosera et Nepenthes.",
    "Météorologie et prévisions climatiques : modèles numériques et incertitudes dans les projections à long terme.",
    "Chimie organique des polymères : synthèse, propriétés mécaniques et applications dans l'industrie moderne.",
    "Géologie des plaques tectoniques : mécanismes de subduction et formation des chaînes de montagnes.",
    "Zoologie des mammifères marins : adaptations physiologiques des cétacés et pinnipèdes à l'environnement aquatique.",
    "Mathématiques fractales : applications dans l'analyse d'images médicales et la modélisation des paysages.",
    "Physique des particules élémentaires : accélérateurs de particules et recherche du boson de Higgs.",
    "Astronomie des exoplanètes : méthodes de détection et caractérisation des planètes extrasolaires.",
    "Paléoclimatologie et ères glaciaires : reconstructions climatiques à partir des carottes de glace.",
    "Biochimie des enzymes : mécanismes catalytiques et ingénierie enzymatique pour applications industrielles.",
    "Géomorphologie des reliefs : processus d'érosion et formation des vallées glaciaires.",
    "Écologie des écosystèmes forestiers : biodiversité, cycles biogéochimiques et services écosystémiques.",
    "Pharmacologie des antidépresseurs : mécanismes d'action moléculaires et effets secondaires neurologiques.",
    "Histoire de l'art roman : architecture religieuse et symbolisme dans l'Europe médiévale.",
    "Anthropologie culturelle : diversité des pratiques sociales et adaptations locales aux environnements.",
    "Sociolinguistique et dialectes : variation linguistique et identité culturelle dans les communautés rurales.",
    "Pédagogie Montessori : principes éducatifs et développement de l'autonomie chez l'enfant.",
    "Épistémologie des sciences : évolution des méthodes scientifiques et impact des paradigmes sur la recherche."
]
BODIES_NEG = [
    "Les volcans islandais représentent une attraction touristique majeure dans le paysage géologique islandais. L'Islande compte plus de 30 systèmes volcaniques actifs, "
    "dont les plus célèbres sont le Eyjafjallajökull et le Grimsvötn. Les éruptions fissurales sont particulièrement fréquentes dans la zone de rift central, "
    "où la dorsale médio-atlantique traverse l'île. Les scientifiques de l'Office Météorologique Islandais surveillent constamment l'activité sismique "
    "et les déformations du sol grâce à un réseau dense de capteurs. Le tourisme géologique génère des revenus importants pour l'économie locale, "
    "attirant des visiteurs du monde entier qui viennent observer des formations de lave récentes et des paysages volcaniques spectaculaires. "
    "Les éruptions de 2010 et 2011 ont eu un impact majeur sur le trafic aérien européen, démontrant l'influence globale de l'activité volcanique islandaise.",

    "L'intelligence artificielle générative révolutionne la création de contenu et la recherche scientifique moderne. Les modèles de langage de grande échelle "
    "comme GPT ont démontré des capacités impressionnantes dans la génération de texte cohérent, la traduction automatique et le résumé de documents complexes. "
    "Les réseaux adverses génératifs (GAN) permettent de créer des images et des vidéos réalistes à partir de descriptions textuelles. L'apprentissage par renforcement "
    "permet aux agents IA de maîtriser des jeux complexes comme les échecs, le Go et les jeux vidéo. Dans le domaine médical, l'IA assistée au diagnostic "
    "améliore la précision des radiologues et accélère la découverte de nouveaux médicaments. Cependant, les enjeux éthiques sont cruciaux : les biais algorithmiques "
    "peuvent perpétuer des discriminations, et la transparence des modèles reste un défi majeur pour la recherche en IA éthique.",

    "The guacamole recipe has evolved significantly since its origins with the Aztecs, who called it ahuacamolli meaning 'avocado sauce'. Traditional Mexican recipes "
    "emphasize the importance of perfectly ripe Hass avocados for optimal texture and flavor profile. Regional variations across Mexico include adding diced tomatoes "
    "in the central regions, jalapeños in the coastal areas, or mango for sweetness in tropical zones. Modern fusion versions incorporate Asian ingredients like "
    "wasabi or ginger, or Mediterranean elements such as feta cheese and olives. Proper preparation technique involves gentle mashing to preserve some chunks "
    "for texture, while the lime juice prevents oxidation and adds acidity. Nutritional analysis reveals high levels of healthy monounsaturated fats, fiber, and "
    "essential vitamins, making guacamole both a delicious and nutritious addition to any meal.",

    "Le tourisme durable représente un défi majeur pour l'industrie du voyage au 21ème siècle. Les principes fondamentaux incluent la minimisation de l'empreinte "
    "carbone à travers l'utilisation de transports à faible émission, le respect des cultures locales et la conservation active de la biodiversité. Les éco-lodges "
    "pionniers utilisent des énergies renouvelables solaires et éoliennes, des matériaux de construction locaux et des systèmes de recyclage des eaux grises. "
    "Les voyageurs responsables privilégient les transports publics, les guides locaux certifiés et les activités respectueuses de l'environnement. La certification "
    "écotourisme, délivrée par des organismes indépendants, garantit le respect des standards environnementaux et sociaux. Les revenus générés par ce type de "
    "tourisme financent directement la protection des parcs nationaux et le développement communautaire local.",

    "Les oiseaux migrateurs font face à des défis sans précédent dus aux changements climatiques anthropiques. Les schémas de migration traditionnels se décalent "
    "de plusieurs semaines, avec des départs plus précoces au printemps et des retours retardés en automne. Certaines espèces comme la sterne arctique étendent "
    "leur aire de répartition vers les pôles, tandis que d'autres comme le traquet motteux voient leur habitat traditionnel se dégrader. La phénologie de reproduction "
    "s'adapte aux nouvelles conditions, mais les populations d'insectes dont dépendent la plupart des oiseaux déclinent rapidement. Les zones humides critiques, "
    "essentielles pour l'escarpe et la reproduction, s'assèchent ou disparaissent sous l'effet de la sécheresse et de l'urbanisation. Les programmes de conservation "
    "internationaux tentent de préserver les corridors migratoires et les sites de nidification essentiels pour la survie de ces espèces.",

    "L'architecture moderne de Tokyo représente une synthèse fascinante entre tradition japonaise et innovation technologique. Les gratte-ciel de Shinjuku et Roppongi "
    "intègrent des éléments de design traditionnel comme les jardins zen et les motifs inspirés de la nature, tout en utilisant des matériaux de pointe comme le verre "
    "intelligent et les structures antisismiques avancées. Le Tokyo Skytree, avec ses 634 mètres, symbolise l'ambition technologique japonaise tout en respectant "
    "les principes de l'harmonie avec l'environnement. Les projets de reconstruction post-Fukushima ont accéléré l'adoption de technologies vertes dans la construction "
    "urbaine. Cette fusion unique d'esthétique traditionnelle et de fonctionnalité moderne fait de Tokyo un laboratoire vivant d'architecture du 21ème siècle.",

    "La Renaissance italienne du 15ème siècle marque un tournant décisif dans l'histoire de l'art et de la science occidentales. Florence, berceau de ce mouvement, "
    "vit naître des génies comme Léonard de Vinci, dont les carnets révèlent une curiosité universelle allant de l'anatomie à l'ingénierie hydraulique, et Michel-Ange, "
    "dont la maîtrise de la sculpture et de la peinture murale révolutionna l'art occidental. La perspective linéaire, développée par Brunelleschi et formalisée par "
    "Alberti, transforma radicalement la représentation de l'espace dans la peinture. Les innovations scientifiques incluaient l'observation astronomique de Copernic "
    "et les dissections anatomiques de Vésale. Cette période d'effervescence intellectuelle, soutenue par les mécènes comme les Médicis, posa les fondements de la "
    "méthode scientifique moderne et de l'humanisme.",

    "Les techniques de méditation zen, originaires du bouddhisme chan chinois, offrent des approches pratiques pour cultiver la pleine conscience et réduire le stress "
    "dans la vie quotidienne moderne. La pratique du zazen, ou méditation assise, développe la concentration et l'observation détachée des pensées. Les études cliniques "
    "randomisées démontrent des effets bénéfiques sur l'anxiété, la dépression et les troubles du sommeil. Les neurosciences cognitives révèlent des modifications "
    "de l'activité cérébrale dans les régions du cortex préfrontal et du système limbique. Au-delà des bienfaits individuels, la méditation zen influence les pratiques "
    "thérapeutiques contemporaines et les programmes de gestion du stress en entreprise.",

    "La technologie blockchain et les cryptomonnaies transforment radicalement les systèmes financiers traditionnels. La blockchain, registre distribué et immuable, "
    "élimine les intermédiaires dans les transactions financières tout en garantissant la transparence et la sécurité. Bitcoin, première cryptomonnaie, a démontré "
    "la viabilité du concept, suivi par Ethereum qui introduisit les contrats intelligents programmables. Les implications pour les systèmes bancaires centralisés "
    "sont profondes : réduction des coûts de transaction, inclusion financière pour les populations non bancarisées, et développement de la finance décentralisée (DeFi). "
    "Les défis incluent la volatilité des prix, la consommation énergétique du minage, et les questions de régulation internationale.",

    "La physique quantique, développée au début du 20ème siècle, révolutionna notre compréhension de la réalité à l'échelle atomique et subatomique. Les concepts "
    "d'intrication quantique, de superposition d'états et d'incertitude de Heisenberg défient l'intuition classique. Les applications pratiques incluent les "
    "ordinateurs quantiques, qui promettent de résoudre des problèmes complexes en cryptographie et en optimisation, et la cryptographie quantique pour des "
    "communications inviolables. L'interprétation de Copenhague, avec son effondrement de la fonction d'onde, reste débattue parmi les physiciens.",

    "Le jardinage urbain et les méthodes de permaculture permettent une production alimentaire durable dans des espaces limités. Les techniques incluent la "
    "culture en lasagne, les buttes autofertiles et les associations de plantes complémentaires. L'amélioration du sol par le compostage et le paillage "
    "réduit la consommation d'eau et favorise la biodiversité microbienne. Dans les environnements urbains, les jardins verticaux, les toitures végétalisées "
    "et les micro-fermes hydroponiques maximisent l'utilisation de l'espace vertical. Ces approches contribuent à la sécurité alimentaire locale et à la "
    "résilience des communautés face aux perturbations climatiques.",

    "L'économie circulaire représente une rupture avec le modèle linéaire traditionnel de production-consommation-déchet. Les systèmes de recyclage innovants "
    "transforment les déchets plastiques en nouveaux matériaux de construction, les déchets organiques en biogaz, et les métaux rares en composants électroniques "
    "reconditionnés. Les entreprises adoptent des modèles de location plutôt que de vente, prolongeant la durée de vie des produits. L'écoconception, qui intègre "
    "la recyclabilité dès la phase de design, réduit l'empreinte environnementale. Les indicateurs de circularité mesurent l'efficacité de ces transitions, "
    "montrant des réductions significatives des émissions de CO2 et de la consommation de ressources.",

    "La recherche en neurosciences révèle comment la plasticité cérébrale influence l'apprentissage et la formation de la mémoire tout au long de la vie. "
    "Les mécanismes de potentialisation à long terme (LTP) et de dépression à long terme (LTD) modulent la force des synapses en réponse à l'activité neuronale. "
    "Chez l'enfant, les périodes critiques de développement permettent l'acquisition rapide du langage et des compétences motrices. Chez l'adulte, la neurogenèse "
    "hippocampique persiste, permettant l'adaptation à de nouveaux environnements. Les applications incluent les programmes de rééducation après accident "
    "vasculaire cérébral et les interventions éducatives optimisées.",

    "La géopolitique du Moyen-Orient implique des interactions complexes entre conflits historiques et dynamiques de pouvoir modernes. Les ressources pétrolières "
    "et gazières influencent les alliances stratégiques, comme en témoignent les relations entre l'Arabie Saoudite, les Émirats Arabes Unis et les puissances "
    "occidentales. Les conflits confessionnels entre sunnites et chiites, exacerbés par la rivalité Iran-Arabie Saoudite, compliquent la stabilité régionale. "
    "Les mouvements djihadistes, de l'État Islamique à Al-Qaïda, exploitent les frustrations sociales et économiques. Les processus de paix israélo-palestiniens "
    "restent dans l'impasse malgré les accords d'Abraham.",

    "La gastronomie moléculaire applique les principes scientifiques à la création d'expériences culinaires innovantes. Les techniques incluent la sphérification, "
    "qui emprisonne des liquides dans des membranes gélifiées, et la cuisson sous vide pour une précision thermique optimale. La chimie des émulsions et des "
    "mousses permet de créer des textures inédites. La physique des transitions de phase explique les changements de texture lors de la cuisson. Ces approches, "
    "pionnières par des chefs comme Ferran Adrià, redéfinissent les frontières entre cuisine, art et science.",

    "Les techniques de photographie de paysage nécessitent une maîtrise de la composition, de l'éclairage et du post-traitement. La règle des tiers guide le "
    "placement des éléments principaux, tandis que les lignes directrices créent de la profondeur. L'heure dorée, juste après le lever ou avant le coucher du soleil, "
    "offre une lumière chaude et directionnelle. Les filtres polarisants réduisent les reflets et saturent les couleurs du ciel. Le post-traitement avec Lightroom "
    "ou Photoshop ajuste l'exposition, le contraste et la balance des blancs. La patience est essentielle pour capturer les conditions météorologiques parfaites.",

    "Les énergies renouvelables marines exploitent les vagues, les marées et les gradients thermiques océaniques pour la production d'électricité. Les systèmes "
    "de conversion de l'énergie des vagues utilisent des flotteurs oscillants ou des colonnes d'eau oscillantes. Les barrages marémoteurs, comme celui de la "
    "Rance en France, captent l'énergie des marées. Les centrales OTEC (Ocean Thermal Energy Conversion) exploitent la différence de température entre surface "
    "et profondeurs. Les évaluations d'impact environnemental examinent les effets sur les écosystèmes marins et les migrations des poissons. Les défis techniques "
    "incluent la corrosion en milieu marin et la variabilité de la ressource.",

    "La sociologie des réseaux sociaux examine comment les plateformes numériques transforment les interactions humaines et la formation de communautés à l'ère digitale. "
    "Les algorithmes de recommandation créent des bulles de filtre, renforçant les opinions existantes. Les mouvements sociaux comme #MeToo ou Black Lives Matter "
    "démontrent le pouvoir mobilisateur des réseaux. La surveillance algorithmique soulève des questions de privacy et de liberté d'expression. Les identités numériques "
    "se construisent à travers les profils, les likes et les partages. Les fake news se propagent rapidement, nécessitant des stratégies de vérification.",

    "L'archéologie précolombienne révèle des civilisations avancées qui prospérèrent en Amérique avant le contact européen. Les Mayas développèrent un système "
    "d'écriture hiéroglyphique complexe, des connaissances astronomiques précises et une architecture monumentale comme Chichen Itza. Les Incas maîtrisèrent "
    "l'ingénierie hydraulique avec des terrasses agricoles et des routes pavées sur des milliers de kilomètres. Les Olmèques créèrent les premières sculptures "
    "colossales et influencèrent les cultures ultérieures. L'agriculture intensive avec le maïs, les haricots et les courges soutint des populations denses. "
    "Ces sociétés démontrent des niveaux de sophistication comparables aux civilisations eurasiennes contemporaines.",

    "La psychologie cognitive étudie les processus mentaux incluant la perception, l'attention, la mémoire et la prise de décision. Les modèles connexionnistes "
    "simulent le traitement de l'information dans les réseaux neuronaux. L'attention sélective filtre les stimuli pertinents dans un environnement surchargé. "
    "La mémoire de travail maintient temporairement l'information pour les tâches cognitives complexes. Les biais cognitifs comme l'effet de confirmation influencent "
    "la prise de décision. Les applications incluent l'interface homme-machine, l'ergonomie des logiciels et les interventions thérapeutiques pour les troubles cognitifs.",

    "L'impact du télétravail sur la productivité organisationnelle a été analysé dans plusieurs études longitudinales. Les changements dans les patterns de "
    "communication incluent une augmentation des emails et des messages instantanés, compensant la réduction des interactions face-à-face. La satisfaction au travail "
    "varie selon les types de tâches : les travaux créatifs bénéficient de la flexibilité, tandis que les tâches collaboratives peuvent souffrir de l'isolement. "
    "Les outils de collaboration numérique comme Slack, Teams et Zoom facilitent la coordination, mais nécessitent une gestion attentive de la charge cognitive. "
    "Les entreprises qui réussissent le télétravail combinent technologies et politiques de bien-être.",

    "Les bienfaits du yoga pour la santé mentale ont été documentés dans de nombreuses études cliniques randomisées. La pratique régulière réduit les niveaux "
    "de cortisol, hormone du stress, et augmente la production d'endorphines. Les techniques de respiration pranayama améliorent la concentration et réduisent "
    "l'anxiété. Les postures physiques (asanas) développent la conscience corporelle et la confiance en soi. La méditation mindfulness, intégrée au yoga, "
    "cultive la pleine conscience et améliore la régulation émotionnelle. Ces effets sont particulièrement bénéfiques pour les populations souffrant de stress "
    "post-traumatique, de dépression et de troubles anxieux.",

    "Les cryptomonnaies et la blockchain redessinent l'avenir de la finance traditionnelle. Bitcoin, créé par Satoshi Nakamoto en 2008, introduisit le concept "
    "de monnaie numérique décentralisée. Ethereum étendit les possibilités avec les contrats intelligents, permettant des applications décentralisées (DApps). "
    "Les stablecoins comme Tether maintiennent une valeur fixe liée aux devises traditionnelles. Les banques centrales explorent les monnaies digitales de "
    "banque centrale (CBDC). Les défis incluent la volatilité, la régulation, et l'impact environnemental du minage. La finance décentralisée (DeFi) propose "
    "des alternatives aux services bancaires traditionnels.",

    "L'intelligence artificielle en médecine transforme le diagnostic, le traitement et la recherche. Les algorithmes de deep learning analysent les images "
    "médicales avec une précision surpassant souvent les radiologues humains pour certaines pathologies. Le machine learning prédit les risques de maladies "
    "chroniques à partir de données génomiques et lifestyle. La découverte de médicaments utilise l'IA pour cribler des millions de molécules candidates. "
    "Les assistants médicaux basés sur l'IA aident à la prise de décision clinique. Les défis éthiques incluent la privacy des données de santé et la validation "
    "clinique des algorithmes.",

    "Le changement climatique impose des adaptations majeures aux pratiques agricoles traditionnelles. Les variétés de cultures résistantes à la sécheresse, "
    "développées par sélection génétique et édition de gènes CRISPR, permettent de maintenir les rendements dans des conditions arides. Les techniques d'irrigation "
    "goutte-à-goutte et de conservation des sols réduisent la consommation d'eau. L'agroforesterie combine cultures et arbres pour améliorer la biodiversité et "
    "la séquestration de carbone. Les systèmes de prévision météorologique aident les agriculteurs à optimiser les dates de semis et de récolte. Ces adaptations "
    "sont cruciales pour la sécurité alimentaire mondiale.",

    "La réalité virtuelle dans l'éducation offre des expériences d'apprentissage immersives et expérientielles. Les simulations historiques permettent aux étudiants "
    "de visiter la Rome antique ou la Révolution française. En sciences, les laboratoires virtuels sécurisent l'apprentissage de réactions chimiques dangereuses. "
    "Les dissections anatomiques virtuelles évitent l'utilisation d'animaux. L'apprentissage des langues étrangères bénéficie d'environnements culturels immersifs. "
    "Les défis incluent le coût des équipements, le mal des transports numériques, et la nécessité de former les enseignants aux nouvelles pédagogies.",

    "Le microbiote intestinal joue un rôle crucial dans le développement et la maturation du système immunitaire. Les bactéries commensales comme les "
    "Bifidobacterium et Lactobacillus colonisent le tractus gastro-intestinal dès la naissance. Elles stimulent la production d'anticorps IgA et modulent "
    "l'inflammation. Les métabolites microbiens, comme les acides gras à chaîne courte, influencent la perméabilité intestinale. Les dysbioses sont associées "
    "à des maladies auto-immunes comme la maladie de Crohn. Les interventions incluent les probiotiques, les prébiotiques et les transplantations de microbiote.",

    "L'évolution des espèces selon Darwin reste observable dans les populations contemporaines. La sélection naturelle agit sur les variations génétiques "
    "favorisant la survie et la reproduction. Les pinsons de Darwin aux Galapagos montrent une adaptation rapide du bec en réponse aux ressources alimentaires "
    "disponibles. Les bactéries développent une résistance aux antibiotiques par mutations et transferts de gènes. La spéciation sympatrique se produit quand "
    "des populations d'une même espèce divergent en l'absence de barrières géographiques. Ces mécanismes démontrent la dynamique continue de l'évolution.",

    "L'écoute de musique classique influence le développement cérébral de l'enfant, selon des études en neurosciences cognitives. Les stimulations auditives "
    "complexes, comme celles de Mozart ou Bach, stimulent la plasticité neuronale dans les aires auditives et frontales. L'effet Mozart, bien que controversé, "
    "suggère une amélioration temporaire des performances spatiales. L'apprentissage musical précoce favorise le développement du langage et des compétences "
    "mathématiques. Les programmes d'éducation musicale dans les écoles montrent des effets positifs sur la concentration et la mémoire de travail.",

    "L'astrophysique des trous noirs révèle des objets cosmiques extrêmes où la gravité déforme l'espace-temps. Les trous noirs stellaires se forment par "
    "l'effondrement gravitationnel d'étoiles massives. Les trous noirs supermassifs, millions de fois plus massifs que le Soleil, résident au centre des galaxies. "
    "L'horizon des événements marque la limite au-delà de laquelle rien ne peut s'échapper. Les ondes gravitationnelles, prédites par Einstein, ont été détectées "
    "provenant de fusions de trous noirs. Les jets relativistes émis par les trous noirs en accrétion alimentent l'activité des quasars et des galaxies actives.",

    "La paléontologie révèle de nouvelles découvertes sur l'extinction du Crétacé, il y a 66 millions d'années. L'impact d'un astéroïde de 10 km de diamètre "
    "dans le golfe du Mexique a créé le cratère de Chicxulub. Les retombées globales ont causé un hiver d'impact, perturbant les chaînes alimentaires. "
    "L'activité volcanique massive des trapps du Deccan a ajouté du CO2 et du SO2 à l'atmosphère. Les dinosaures non-aviens disparurent, tandis que les ancêtres "
    "des oiseaux modernes survécurent. Ces événements montrent comment des perturbations environnementales rapides peuvent causer des extinctions massives.",

    "La génétique moléculaire et les technologies de séquençage de nouvelle génération révolutionnent le diagnostic et le traitement des maladies héréditaires. "
    "Les thérapies géniques, utilisant des vecteurs viraux comme AAV, corrigent les mutations causales dans des maladies comme l'amyotrophie spinale. L'édition "
    "de gènes CRISPR-Cas9 permet des modifications précises du génome. Les tests génétiques préventifs identifient les risques de cancer du sein ou d'Alzheimer. "
    "Les bases de données génomiques comme gnomAD fournissent des références pour l'interprétation des variants. Les défis éthiques incluent la thérapie germinale "
    "et l'accès équitable aux technologies.",

    "L'océanographie révèle comment les gyres océaniques influencent le climat global et la distribution des espèces marines. Les courants de surface, comme "
    "le Gulf Stream, transportent la chaleur des tropiques vers les pôles. Les tourbillons de mesoscale affectent la productivité biologique. Les zones de "
    "convergence comme l'Atlantique Nord accumulent les plastiques flottants. Les changements climatiques modifient l'intensité et les trajectoires des courants, "
    "avec des impacts sur les pêcheries et la météorologie côtière. Les modèles numériques prédisent une accélération de ces changements.",

    "La botanique des plantes carnivores révèle des adaptations morphologiques et physiologiques fascinantes. Les Drosera utilisent des tentacules collants "
    "pour capturer les insectes, tandis que les Nepenthes développent des urnes remplies de liquide digestif. Les enzymes comme les phosphatases et les "
    "protéases décomposent les proies. Ces adaptations évoluèrent dans des sols pauvres en nutriments, comme les tourbières acides. Les mécanismes de mouvement "
    "rapide des Dionaea (attrape-mouches) impliquent des changements de turgescence cellulaire. Ces plantes démontrent l'ingéniosité évolutive face aux contraintes "
    "environnementales.",

    "La météorologie et les modèles de prévision climatique font face à des incertitudes croissantes dues au changement climatique. Les équations de Navier-Stokes "
    "décrivent les mouvements atmosphériques, mais leur résolution numérique nécessite des approximations. Les modèles couplés océan-atmosphère comme ceux du GIEC "
    "projettent un réchauffement de 1.5 à 4°C d'ici 2100 selon les scénarios d'émission. Les incertitudes incluent la sensibilité climatique, les rétroactions "
    "des nuages, et les cycles biogéochimiques. L'assimilation de données satellitaires améliore les prévisions à court terme.",

    "La chimie organique des polymères explore la synthèse, les propriétés mécaniques et les applications industrielles. Les polymères thermoplastiques comme le "
    "polyéthylène sont produits par polymérisation radicalaire ou ionique. Les élastomères comme le caoutchouc naturel possèdent des propriétés élastiques dues "
    "à leur structure amorphe. Les polymères conducteurs comme le polyacétylène trouvent des applications en électronique organique. La dégradation des plastiques "
    "pose des défis environnementaux, stimulant la recherche en polymères biodégradables.",

    "La géologie des plaques tectoniques explique la dynamique de la surface terrestre. La subduction des plaques océaniques sous les continents crée des zones "
    "de volcanisme et de séismes, comme la ceinture de feu du Pacifique. La divergence au niveau des dorsales médio-océaniques génère de la croûte océanique "
    "nouvelle. Les collisions continentales forment des chaînes de montagnes comme l'Himalaya. Les points chauds mantelliques, comme sous Hawaï, créent des "
    "chaînes d'îles volcaniques. La théorie de la tectonique des plaques, développée dans les années 1960, révolutionna la géologie.",

    "La zoologie des mammifères marins révèle des adaptations physiologiques remarquables à l'environnement aquatique. Les cétacés, descendants des artiodactyles "
    "terrestres, développèrent une hydrodynamique parfaite avec des nageoires et une queue puissante. Les pinnipèdes comme les phoques possèdent une fourrure "
    "imperméable et une couche de graisse isolante. La plongée profonde nécessite des adaptations cardiovasculaires : bradycardie et redistribution du sang vers "
    "les organes vitaux. L'écholocalisation chez les dauphins et les chauves-souris démontre une convergence évolutive. La pollution marine menace ces espèces "
    "sensibles.",

    "Les mathématiques fractales trouvent des applications dans l'analyse d'images médicales et la modélisation des paysages. Les fractales auto-similaires, "
    "comme l'ensemble de Mandelbrot, possèdent une dimension fractale non entière. En imagerie médicale, les fractales analysent la texture des tissus pathologiques. "
    "En géographie, les fractales modélisent l'érosion côtière et la distribution des rivières. L'algorithme de diamant-carré génère des terrains réalistes pour "
    "les simulations. La théorie des fractales révèle l'auto-organisation dans les systèmes complexes.",

    "La physique des particules élémentaires explore la matière à ses échelles les plus fondamentales. Les accélérateurs comme le LHC au CERN font collisionner "
    "des protons à des énergies de 13 TeV, produisant des particules éphémères. La découverte du boson de Higgs en 2012 confirma le mécanisme de Brout-Englert-Higgs "
    "pour la masse des particules. Le modèle standard décrit trois forces fondamentales et les quarks, leptons et bosons de jauge. La recherche de nouvelle physique "
    "inclut la supersymétrie et la matière noire. Les neutrinos, particules insaisissables, oscillent entre différents types.",

    "L'astronomie des exoplanètes utilise des méthodes indirectes pour détecter et caractériser les planètes en dehors de notre système solaire. La méthode des "
    "transits mesure la baisse de luminosité de l'étoile lors du passage de la planète. La vitesse radiale détecte le mouvement de l'étoile due à la gravitation "
    "planétaire. Plus de 5000 exoplanètes ont été découvertes, révélant une diversité de tailles, compositions et orbites. La zone habitable, où l'eau liquide "
    "peut exister, guide la recherche de vie extraterrestre. Les télescopes spatiaux comme JWST caractérisent les atmosphères exoplanétaires.",

    "La paléoclimatologie utilise les carottes de glace pour reconstruire les climats passés. Les bulles d'air emprisonnées dans la glace antarctique contiennent "
    "des échantillons d'atmosphère datant de 800 000 ans. Les isotopes de l'oxygène (δ18O) indiquent les températures passées. Les ères glaciaires, espacées "
    "de 100 000 ans, sont liées aux cycles de Milankovitch. Les carottes sédimentaires océaniques révèlent des changements plus anciens. Les reconstructions "
    "paléoclimatiques aident à valider les modèles climatiques et à prédire les changements futurs.",

    "La biochimie des enzymes révèle les mécanismes catalytiques qui accélèrent les réactions biologiques. Les enzymes abaissent l'énergie d'activation par "
    "stabilisation du complexe enzyme-substrat. Les sites actifs contiennent souvent des métaux comme le zinc ou le fer. L'ingénierie enzymatique modifie "
    "les enzymes pour des applications industrielles : dégradation des plastiques, synthèse de médicaments, production de biocarburants. La biologie structurale "
    "utilise la cristallographie aux rayons X pour déterminer les structures tridimensionnelles.",

    "La géomorphologie étudie les processus d'érosion et de formation des reliefs terrestres. L'érosion hydraulique creuse les vallées fluviales, tandis que "
    "l'érosion éolienne façonne les déserts et les dunes. Les glaciers creusent les vallées en U et déposent des moraines. Les processus de météorisation "
    "décomposent les roches en sols. L'élévation tectonique et l'érosion interagissent pour maintenir l'équilibre dynamique des paysages. Les modèles numériques "
    "simulent l'évolution des bassins versants sur des échelles temporelles géologiques.",

    "L'écologie des écosystèmes forestiers examine la biodiversité, les cycles biogéochimiques et les services écosystémiques. Les forêts tropicales abritent "
    "50% de la biodiversité terrestre malgré ne couvrant que 7% des terres émergées. Les cycles du carbone, de l'azote et du phosphore maintiennent la "
    "productivité. La déforestation perturbe ces équilibres, libérant du CO2 stocké et réduisant la biodiversité. Les services écosystémiques incluent la "
    "régulation du climat, la purification de l'eau et la pollinisation. La gestion durable vise à concilier exploitation et conservation.",

    "La pharmacologie des antidépresseurs explore les mécanismes d'action moléculaires et les effets secondaires neurologiques. Les inhibiteurs sélectifs de la "
    "recapture de la sérotonine (ISRS) augmentent la disponibilité de sérotonine dans la fente synaptique. Les effets secondaires incluent nausées, insomnies "
    "et dysfonctions sexuelles. La variabilité interindividuelle dans la réponse thérapeutique est liée aux polymorphismes génétiques du cytochrome P450. "
    "Les nouveaux antidépresseurs ciblent des voies comme le glutamate et les neurotrophines.",

    "L'histoire de l'art roman révèle l'architecture religieuse et le symbolisme dans l'Europe médiévale du 11ème et 12ème siècle. Les églises romanes, avec "
    "leurs voûtes en berceau et leurs arcs en plein cintre, symbolisent la solidité de la foi chrétienne. Les chapiteaux historiés racontent des scènes bibliques "
    "et des fables morales. Le pèlerinage vers Saint-Jacques-de-Compostelle stimula la construction d'églises le long des chemins. Les monastères comme Cluny "
    "influencèrent l'architecture religieuse. L'art roman précéda le gothique, avec sa recherche de lumière et de hauteur.",

    "L'anthropologie culturelle explore la diversité des pratiques sociales et les adaptations locales aux environnements. Les sociétés de chasseurs-cueilleurs "
    "comme les San du Kalahari développèrent une connaissance intime de leur environnement. Les sociétés agraires créèrent des calendriers complexes basés sur "
    "les cycles lunaires et solaires. Les rituels de passage marquent les transitions de la vie. Les systèmes de parenté régulent les alliances matrimoniales. "
    "La mondialisation homogénéise certaines pratiques tout en ravivant les identités locales.",

    "La sociolinguistique examine la variation linguistique et l'identité culturelle dans les communautés rurales. Les dialectes régionaux préservent des "
    "archaïsmes linguistiques et reflètent l'histoire des migrations. L'identité linguistique renforce le sentiment d'appartenance communautaire. Les attitudes "
    "linguistiques influencent la transmission intergénérationnelle des langues. Les langues minoritaires font face à la pression de l'anglais global. "
    "La revitalisation linguistique utilise l'éducation et les médias pour préserver le patrimoine linguistique.",

    "La pédagogie Montessori, développée par Maria Montessori au début du 20ème siècle, met l'accent sur le développement de l'autonomie chez l'enfant. "
    "L'environnement préparé offre des matériaux auto-correctifs qui permettent l'apprentissage par l'expérience. Les périodes sensibles correspondent aux "
    "moments optimaux pour l'acquisition de compétences spécifiques. L'éducation mixte d'âges favorise l'entraide et l'empathie. La liberté de choix dans "
    "les activités respecte le rythme individuel de chaque enfant. Les résultats incluent une meilleure concentration et une plus grande confiance en soi.",

    "L'épistémologie des sciences examine l'évolution des méthodes scientifiques et l'impact des paradigmes sur la recherche. La révolution copernicienne "
    "remplaça le géocentrisme par l'héliocentrisme. La méthode hypothético-déductive de Popper insiste sur la falsifiabilité. Les révolutions scientifiques, "
    "selon Kuhn, impliquent des changements de paradigmes. La science post-normale traite des questions à forts enjeux sociétaux. La reproductibilité des "
    "expériences reste un défi dans de nombreux domaines."
]

# -------------------------- génération d'articles --------------------------

def mk_positive(i: int) -> Dict[str, str]:
    lang_fr = with_prob(0.6)  # plus de FR que EN
    if lang_fr:
        title = random.choice(TITLES_POS_FR)
        abstract = random.choice(ABSTRACTS_POS_FR)
        body = random.choice(BODIES_POS_FR)
        journal = random.choice(JOURNALS_FR)
        author = random.choice(AUTHORS_FR)
        lang = "fr"
    else:
        title = random.choice(TITLES_POS_EN)
        abstract = random.choice(ABSTRACTS_POS_EN)
        body = random.choice(BODIES_POS_EN)
        journal = random.choice(JOURNALS_EN)
        author = random.choice(AUTHORS_EN)
        lang = "en"

    # bruit contrôlé (augmenté)
    if with_prob(0.4):  # Plus de HTML
        abstract = inject_complex_html_noise(abstract)
    if with_prob(0.3):  # Plus de HTML complexe
        body = inject_complex_html_noise(body)
    if with_prob(0.15):
        title = typo_perturb(title)
    if with_prob(0.2):  # Plus de unicode
        abstract = inject_unicode_noise(abstract)
    if with_prob(0.15):
        body = inject_unicode_noise(body)

    # quelques abstracts trop courts (pour tester le filtre min_abstract_len)
    if with_prob(0.12):  # Plus fréquent
        abstract = "Étude préliminaire." if lang == "fr" else "Preliminary note."

    # Articles très longs (pour tester max_text_len)
    if with_prob(0.05):
        body = body * 3  # Triple la longueur

    url = messy_url("https://example.com/cat_light", i, themed=True)
    date = rand_date().date().isoformat()
    doi = f"10.1234/cats.{random.randint(1000,9999)}.{i}" if with_prob(0.2) else ""

    return {
        "url": url,
        "title": title,
        "abstract": abstract,
        "body": body,
        "lang_hint": lang,
        "author": author,
        "journal": journal,
        "published_at": date,
        "doi": doi,
        "quality_type": "normal"
    }

def mk_negative(i: int) -> Dict[str, str]:
    title = random.choice(TITLES_NEG)
    abstract = random.choice(ABSTRACTS_NEG)
    body = random.choice(BODIES_NEG)
    # bruit (augmenté)
    if with_prob(0.3):  # Plus de HTML
        abstract = inject_complex_html_noise(abstract)
    if with_prob(0.2):  # Plus de HTML
        body = inject_complex_html_noise(body)
    if with_prob(0.15):
        title = typo_perturb(title)
    if with_prob(0.2):  # Plus de unicode
        abstract = inject_unicode_noise(abstract)
    if with_prob(0.15):
        body = inject_unicode_noise(body)

    # langues non FR/EN (petit %), utiles pour tester langdetect
    if with_prob(0.08):  # Plus fréquent
        abstract = "Die Vulkane Islands sind spektakulär."
    if with_prob(0.05):
        abstract = "火山は観光の目玉となっている。"
    if with_prob(0.03):
        body = "这是一个关于其他主题的中文文章。" * 15

    # Articles très courts ou très longs
    if with_prob(0.06):
        abstract = "Short."
    if with_prob(0.04):
        body = body * 2

    url = messy_url("https://example.com/random", i, themed=False)
    date = rand_date().date().isoformat()
    return {
        "url": url,
        "title": title,
        "abstract": abstract,
        "body": body,
        "lang_hint": "",
        "author": random.choice(AUTHORS_FR + AUTHORS_EN),
        "journal": random.choice(JOURNALS_FR + JOURNALS_EN),
        "published_at": date,
        "doi": "",
        "quality_type": "off_topic"
    }

def make_exact_duplicate(row: Dict[str, str], i: int) -> Dict[str, str]:
    r = dict(row)
    # seule l'URL change (mais de façon "sale") pour tester la dédup URL/titre
    r["url"] = messy_url("https://www.example.com/cat_light", i, themed=True)
    r["quality_type"] = "exact_duplicate"
    if with_prob(0.5):
        r["title"] = r["title"].strip() + " "  # espace traînant, même titre
    return r

def make_near_duplicate(row: Dict[str, str], i: int) -> Dict[str, str]:
    r = dict(row)
    r["url"] = messy_url("https://example.com/cat_light", i, themed=True)
    r["title"] = near_duplicate_text(r["title"])
    r["abstract"] = near_duplicate_text(r["abstract"])
    r["quality_type"] = "near_duplicate"
    if with_prob(0.5):
        r["body"] = near_duplicate_text(r["body"])
    return r

# -------------------------- programme principal --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=200, help="Nombre d'articles thème 'chat ✕ lumière'")
    ap.add_argument("--n-neg", type=int, default=150, help="Nombre d'articles hors sujet")
    ap.add_argument("--n-dupes", type=int, default=25, help="Nombre de doublons exacts du thème")
    ap.add_argument("--n-near", type=int, default=40, help="Nombre de quasi-doublons du thème")
    ap.add_argument("--n-multilang", type=int, default=30, help="Nombre d'articles multilingues")
    ap.add_argument("--n-low-quality", type=int, default=20, help="Nombre d'articles de faible qualité")
    ap.add_argument("--seed", type=int, default=42, help="Graine de random")
    ap.add_argument("--out", type=str, default="data/articles_fictifs.csv", help="Chemin du CSV de sortie")
    args = ap.parse_args()

    random.seed(args.seed)

    # corpus de base
    pos = [mk_positive(i) for i in range(args.n_pos)]
    neg = [mk_negative(i) for i in range(args.n_neg)]

    # articles de faible qualité
    low_quality = []
    for i in range(args.n_low_quality):
        lang = random.choice(["fr", "en"])
        low_quality.append(create_low_quality_article(lang))

    # articles multilingues
    multilingual = []
    for i in range(args.n_multilang):
        multilingual.append(create_multilingual_article(10000 + i))

    # choisit des positifs pour fabriquer des doublons/quasi-doublons
    pick_for_dupes = random.sample(pos, k=min(args.n_dupes, max(1, len(pos)//3))) if args.n_dupes > 0 else []
    pick_for_near  = random.sample(pos, k=min(args.n_near,  max(1, len(pos)//2))) if args.n_near  > 0 else []

    dupes = []
    near  = []
    for idx, row in enumerate(pick_for_dupes):
        dupes.append(make_exact_duplicate(row, 1000 + idx))
    for idx, row in enumerate(pick_for_near):
        near.append(make_near_duplicate(row, 2000 + idx))

    # mélange et sortie
    rows: List[Dict[str, str]] = pos + neg + low_quality + multilingual + dupes + near
    random.shuffle(rows)

    # s'assurer des colonnes minimales
    base_cols = ["url", "title", "abstract", "body"]
    extra_cols = ["lang_hint", "author", "journal", "published_at", "doi", "quality_type"]
    all_cols = list(base_cols + extra_cols)

    df = pd.DataFrame(rows, columns=all_cols)

    # Remplir les valeurs manquantes pour quality_type
    df["quality_type"] = df.get("quality_type", "normal").fillna("normal")

    df.to_csv(args.out, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    print(f"[OK] Fichier '{args.out}' genere avec {len(df)} articles.")
    print(f"    - {len(pos)} articles pertinents (theme chat+lumiere)")
    print(f"    - {len(neg)} articles hors-sujet")
    print(f"    - {len(low_quality)} articles de faible qualite")
    print(f"    - {len(multilingual)} articles multilingues")
    print(f"    - {len(dupes)} doublons exacts")
    print(f"    - {len(near)} quasi-doublons")
    print(f"    - Total: {len(df)} articles")


if __name__ == "__main__":
    main()
