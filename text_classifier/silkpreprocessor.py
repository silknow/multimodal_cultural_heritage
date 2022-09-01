#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

"""
Text preprocessing functions.
"""
import unicodedata
import re
from functools import reduce, partial

import ftfy
from sacremoses import MosesTokenizer
from unidecode import unidecode

#
# CONSTANTS
#

RPLS = {
    "ssilk": "silk",
    "inches": "inches",
    "length": "length",
    "jaqard": "jacquard",
    "jaquard": "jacquard",
    "italiantextile": "italian textile",
    "needleweaving": "needle weaving",
    "waistseam": "waist seam",
    "sleeveband": "sleeve band",
    "drawnwork": "drawn work",
    "needlework": "needle work",
    "needlewoven": "needle woven",
    "threadwork": "thread work",
    "needlecase": "needle case",
    "longsleeve": "long sleeve",
    "designerembroidery": "designer embroidery",
    "lampàs": "lampas",
    "esfumaturas": "esfumado",
    "madrids": "madrid",
    "botehs": "boteh",
    "halbedier": "halberdier",
    "manuscruipt": "manuscript",
    "latchets": "latchet",
    "lustring": "calendering",
    "unplied": "not plied",
    "cannellé": "canelle",
    "canellé": "canelle",
    "clothiing": "clothing",
    "bizantinos": "byzantine",
    "backseam": "back seam",
    "unembroidered": "not embroidered",
    "emboidered": "embroidered",
    "floorspread": "floor spread",
    "overknit": "over knit",
    "overstitch": "over stitch",
    "underbodice": "under bodice",
    "undersleeve": "under sleeve",
    "handscreens": "hand screens",
    "backstitched": "back stitched",
    "regiion": "region",
    "lisere": "edging",
    "laceing": "lacing",
    "commmission": "commission",
    "espolinadas": "brocado",
    "espolinada": "brocado",
    "espolinado": "brocado",
    "brochadas": "brocado",
    "brochada": "brocado",
    "esfumaturas": "esfumado",
    "esfumatura": "esfumado",
    "lampàs": "lampas",
    "éventails": "eventail",
    "beentjes": "beentje",
    "abanos": "abano",
    "espolinadas": "brocado",
    "espolinades": "brocado",
    "espolinat": "brocado",
    "espolinada": "brocado",
    "brochadas": "brocado",
    "lampàs": "lampas",
    "intensidad": "intensitat",
}

REP_RGX = []
for ori, rpl in RPLS.items():
    orig = " " + ori + " "
    rgx = re.compile(re.escape(ori), re.IGNORECASE)
    REP_RGX.append(rgx)


cleanr = re.compile("<.*?>")
cleanr_e = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def fix_text(text):
    """Fixes text issues using ftfy including mojibake and html entities.
    Returns NFC normalized unicode text."""
    return ftfy.fix_text(text)


def clean_html(s):
    """Uses a regular expression to remove leftover html tags in text."""
    s = re.sub(cleanr, "", s)
    s = re.sub(cleanr_e, "", s)
    return s


def replace_string(text):
    """A wrapper around str.replace where replacements is a dictionary:
        original_string -> replacement_string
    whitespace=True surounds the replacement with whitespaces.
    """
    for rgx in REP_RGX:
        txt = rgx.sub(rpl, text)
    return text


def sn_preprocess_text(text, replace_strings=True):
    """SilkNOW: preprocess and tokenize text.
    Returns preprocessed and tokenized text as a space delimited sequence of
    tokens (i.e. returns a string).
    """

    text = clean_html(text)
    if replace_strings:
        text = replace_string(text)

    return text
