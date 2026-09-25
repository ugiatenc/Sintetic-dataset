#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sales i entorns acustics propis de la Val d'Aran per al simulador."""

from __future__ import annotations

import sys
from pathlib import Path

ARREL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARREL / "src"))

import acoustic_sim as ac
from acoustic_sim import Entorn, Sala

AMBIENTS_ARANES = {
    "riu":            {"demand": "NRIVER",   "split": "benchmark_test",
                       "descripcio": "Riu de muntanya: aigua continua, banda ampla"},
    "camp":           {"demand": "NFIELD",   "split": "feature_scorer_train",
                       "descripcio": "Camp obert: vent, ocells, res de trafic"},
    "parc":           {"demand": "NPARK",    "split": "scorer_val",
                       "descripcio": "Parc: gent llunyana, ocells, vent suau"},
    "cotxe":          {"demand": "TCAR",     "split": "feature_scorer_train",
                       "descripcio": "Interior de cotxe en marxa: motor i rodament"},
    "autobus":        {"demand": "TBUS",     "split": "scorer_val",
                       "descripcio": "Interior d'autobus: motor, vibracio, gent"},
    "casa":           {"demand": "DLIVING",  "split": "feature_scorer_train",
                       "descripcio": "Sala d'estar: electrodomestics llunyans, carrer de fons"},
    "cuina":          {"demand": "DKITCHEN", "split": "feature_scorer_train",
                       "descripcio": "Cuina: aixeta, estris, campana"},
    "passadis_escola": {"demand": "OHALLWAY", "split": "benchmark_test",
                       "descripcio": "Passadis d'edifici public: passes, portes, veus llunyanes"},
}

SALES_ARANES = {
    "glesia": Sala("glesia", (28.0, 11.0, 13.0), 2.6, 4.0,
                   "Esglesia de pedra: nau alta, cua de reverberacio molt llarga, micro lluny"),
    "sala_plens": Sala("sala_plens", (14.0, 9.0, 4.5), 1.0, 2.5,
                       "Sala de plens del Conselh: sostre alt, pedra i fusta, micro de taula"),
    "ajuntament": Sala("ajuntament", (6.0, 5.0, 3.0), 0.5, 1.0,
                       "Despatx d'ajuntament petit: parets nues, poca absorcio"),
    "aula": Sala("aula", (9.0, 7.0, 3.0), 0.7, 2.0,
                 "Aula: superficies dures, micro a distancia mitjana"),
    "menjador": Sala("menjador", (5.0, 4.0, 2.6), 0.35, 0.8,
                     "Menjador de casa: moblat, absorcio alta, micro a prop"),
    "estudi_radio": Sala("estudi_radio", (4.0, 3.0, 2.5), 0.2, 0.3,
                         "Estudi de radio local: tractat, micro molt a prop"),
}

ENTORNS_ARANES = {
    "estudi_radio": Entorn(
        "estudi_radio", "Estudi de Radio Aran: sala tractada i micro a prop; la condicio neta de referencia",
        pes=2.5, sala="estudi_radio", ambient=None, snr_db=None, canal="estudi",
        styles=("conselharan", "arannoticies", "institut", "claude", "bsc", "aina", "wikipedia")),
    "plens_conselh": Entorn(
        "plens_conselh", "Ple del Conselh Generau: sala de pedra, murmuri contingut, micro de taula",
        pes=1.5, sala="sala_plens", ambient="sala_premsa", snr_db=(18.0, 28.0), canal="micro_faristol",
        styles=("conselharan", "institut")),
    "ajuntament": Entorn(
        "ajuntament", "Declaracions a un ajuntament petit: despatx nu, micro de corbata",
        pes=1.2, sala="ajuntament", ambient="redaccio", snr_db=(16.0, 26.0), canal="micro_corbata",
        styles=("conselharan", "arannoticies")),
    "glesia_acte": Entorn(
        "glesia_acte", "Acte en una esglesia: reverberacio de pedra molt llarga, micro de faristol",
        pes=0.6, sala="glesia", ambient="sala_premsa", snr_db=(22.0, 32.0), canal="micro_faristol",
        styles=("conselharan", "institut", "claude")),
    "aula": Entorn(
        "aula", "Aula d'escola: reverberacio mitjana i passadis de fons",
        pes=0.8, sala="aula", ambient="passadis_escola", snr_db=(14.0, 24.0), canal="micro_corbata",
        styles=("institut", "claude", "conselharan")),
    "casa": Entorn(
        "casa", "Lectura a casa: menjador moblat, ambient domestic suau",
        pes=1.2, sala="menjador", ambient="casa", snr_db=(12.0, 22.0), canal="micro_corbata",
        styles=("bsc", "aina", "wikipedia", "claude")),
    "cuina": Entorn(
        "cuina", "Conversa a la cuina pel mobil: estris i aixeta a prop",
        pes=0.5, sala="menjador", ambient="cuina", snr_db=(8.0, 18.0), canal="mobil_voip",
        styles=("bsc", "claude")),
    "muntanya_riu": Entorn(
        "muntanya_riu", "Cronica a peu de riu: aigua continua, sense sala, micro de reporter",
        pes=1.0, sala=None, ambient="riu", snr_db=(6.0, 16.0), canal="micro_corbata",
        styles=("arannoticies", "claude", "conselharan")),
    "muntanya_vent": Entorn(
        "muntanya_vent", "Exterior de muntanya: vent i camp obert, micro de reporter",
        pes=1.0, sala=None, ambient="camp", snr_db=(8.0, 18.0), canal="micro_corbata",
        styles=("arannoticies", "claude", "conselharan")),
    "carretera_cotxe": Entorn(
        "carretera_cotxe", "Connexio des del cotxe a la carretera del port: motor i mobil",
        pes=0.8, sala=None, ambient="cotxe", snr_db=(8.0, 16.0), canal="mobil_voip",
        styles=("arannoticies", "claude")),
    "autobus": Entorn(
        "autobus", "Dins de l'autobus de linia: motor, vibracio, gent",
        pes=0.4, sala=None, ambient="autobus", snr_db=(6.0, 14.0), canal="mobil_voip",
        styles=("claude", "bsc")),
    "telefon_local": Entorn(
        "telefon_local", "Trucada a la radio local des de casa: banda telefonica",
        pes=1.0, sala="menjador", ambient="casa", snr_db=(12.0, 20.0), canal="telefon",
        styles=("arannoticies", "claude", "conselharan")),
    "parc": Entorn(
        "parc", "Entrevista al parc del poble: ocells, gent llunyana",
        pes=0.6, sala=None, ambient="parc", snr_db=(12.0, 22.0), canal="micro_corbata",
        styles=("arannoticies", "claude")),
}

PESOS_GENERICS = {"estudi_net": 1.0, "telefon": 0.6, "carrer": 0.5, "exterior_gent": 0.5,
                  "sala_premsa": 0.6, "cafeteria": 0.4,
                  "plato_tv": 0.0, "redaccio": 0.0, "sala_actes": 0.0, "connexio_movil": 0.0,
                  "font_real": 0.0}


def registrar() -> None:
    """Fica sales i entorns d'Aran a les taules del simulador, que es on el mesclador els busca."""
    ac.SALES.update(SALES_ARANES)
    ac.ENTORNS.update(ENTORNS_ARANES)
    for nom, pes in PESOS_GENERICS.items():
        if nom in ac.ENTORNS:
            ac.ENTORNS[nom].pes = pes


registrar()
ENTORNS_ACTIUS = {k: e for k, e in ac.ENTORNS.items() if e.pes > 0}
