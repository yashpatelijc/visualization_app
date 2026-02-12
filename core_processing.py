#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core_processing.py

Features:
  1) Excel-style expression parser (backup).
  2) condition_to_expr + evaluate_condition_structured (structured conditions).
  3) Helper functions: compute_offset, compute_entry_size_if_currency, compute_target_stop.
  4) apply_trade_setup_df (normal simulation):
       - Start/continue logic for (Long/Short × High/Low).
       - Per-stop & per-target "allow reentry" toggles (initial & reentry).
       - Entry-level trailing stop (entry_most_fav_price) with reentry toggle.
       - Global trailing stop (disallows future entries).
       - Freed-slot logic for reentries.
       - **Improved** approach: param fields (Stop_Method, Target_Method, etc.) are stored
         both in the local e_dict AND in the corresponding "New Trend Entry"/"Reentry"
         event dictionary. This way, get_normalized_trades can retrieve them directly.
  5) simulate_variant: simplified ATR-based approach ignoring advanced trailing stops.
  6) get_normalized_trades: final trade list with extra columns for param fields.
"""

import pandas as pd
import numpy as np
import logging
import streamlit as st  # not strictly necessary, but included for consistency

# -----------------------------------------------------------------------------
# 1) Excel-Style Expression Parser
# -----------------------------------------------------------------------------
from pyparsing import (
    Literal, Word, alphanums, nums, Combine,
    CaselessKeyword, oneOf, infixNotation, opAssoc, Group, Forward, ParserElement, delimitedList
)
ParserElement.enablePackrat()

def if_func(cond, true_val, false_val):
    return true_val if cond else false_val

def flatten_tokens(tokens):
    if isinstance(tokens, str):
        return tokens
    elif isinstance(tokens, (list, tuple)):
        return " ".join(flatten_tokens(tok) for tok in tokens)
    else:
        return str(tokens)

def if_action(tokens):
    cond = flatten_tokens(tokens[1])
    true_expr = flatten_tokens(tokens[2])
    false_expr = flatten_tokens(tokens[3])
    return f"if({cond}, {true_expr}, {false_expr})"

def and_action(tokens):
    args = tokens[1].asList()
    flat_args = ", ".join(flatten_tokens(arg) for arg in args)
    return f"all([{flat_args}])"

def or_action(tokens):
    args = tokens[1].asList()
    flat_args = ", ".join(flatten_tokens(arg) for arg in args)
    return f"any([{flat_args}])"

identifier = Word("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_", alphanums + "_")
integer = Word(nums)
floatnumber = Combine(Word(nums) + "." + Word(nums))
number = floatnumber | integer
true_literal = CaselessKeyword("TRUE").setParseAction(lambda _: "True")
false_literal = CaselessKeyword("FALSE").setParseAction(lambda _: "False")
lpar = Literal("(").suppress()
rpar = Literal(")").suppress()
comma = Literal(",").suppress()
expr = Forward()

if_expr = (CaselessKeyword("IF") + lpar + Group(expr) + comma + Group(expr) + comma + Group(expr) + rpar)
if_expr.setParseAction(if_action)

and_expr = (CaselessKeyword("AND") + lpar + Group(delimitedList(expr)) + rpar)
and_expr.setParseAction(and_action)

or_expr = (CaselessKeyword("OR") + lpar + Group(delimitedList(expr)) + rpar)
or_expr.setParseAction(or_action)

operand = if_expr | and_expr | or_expr | number | true_literal | false_literal | identifier | (lpar + expr + rpar)
expr <<= infixNotation(operand, [
    (oneOf("* /"), 2, opAssoc.LEFT, lambda t: flatten_tokens(t.asList())),
    (oneOf("+ -"), 2, opAssoc.LEFT, lambda t: flatten_tokens(t.asList())),
    (oneOf(">= <= > < == !="), 2, opAssoc.LEFT, lambda t: flatten_tokens(t.asList())),
])

def parse_excel_to_python(expr_str):
    try:
        parsed = expr.parseString(expr_str, parseAll=True)[0]
        python_expr = flatten_tokens(parsed)
        logging.info("Parsed expression: %s", python_expr)
        return python_expr
    except Exception as e:
        logging.error("Failed to parse expression '%s': %s", expr_str, e)
        return expr_str

def evaluate_condition(expr_str, row):
    python_expr = parse_excel_to_python(expr_str)
    safe_globals = {"if": if_func, "all": all, "any": any}
    safe_locals = row.to_dict()
    try:
        result = eval(python_expr, safe_globals, safe_locals)
    except Exception as e:
        logging.error("Error evaluating '%s' on row %s: %s", python_expr, row.to_dict(), e)
        result = False
    return bool(result)

# -----------------------------------------------------------------------------
# 2) condition_to_expr + evaluate_condition_structured
# -----------------------------------------------------------------------------
def condition_to_expr(condition):
    if condition["type"]=="simple":
        left = str(condition["left_value"]) if condition["left_type"]=="constant" else condition["left_value"]
        right= str(condition["right_value"]) if condition["right_type"]=="constant" else condition["right_value"]
        return f"({left} {condition['operator']} {right})"
    elif condition["type"]=="group":
        exprs = [condition_to_expr(c) for c in condition["conditions"]]
        joiner= " and " if condition["logic"]=="AND" else " or "
        return "(" + joiner.join(exprs)+ ")"
    else:
        return ""

def evaluate_condition_structured(condition, row):
    expr_str = condition_to_expr(condition)
    safe_globals={}
    safe_locals= row.to_dict()
    try:
        result= eval(expr_str, safe_globals, safe_locals)
    except Exception as e:
        logging.error("Error evaluating structured condition '%s': %s", expr_str, e)
        result=False
    return bool(result)

def get_condition_group(key_prefix, indicator_list, default=None):
    st.markdown(f"**Condition Group: {key_prefix}**")
    default_logic = default.get("logic","AND") if default else "AND"
    default_n = default.get("n",1) if default else 1
    default_conditions = default.get("conditions",[]) if default else []
    logic = st.selectbox(f"{key_prefix} - Logic", ["AND","OR"], index=["AND","OR"].index(default_logic.upper()), key=f"{key_prefix}_logic")
    n = st.number_input(f"{key_prefix} - # Conditions", min_value=1, max_value=5, value=default_n, key=f"{key_prefix}_n")
    conditions=[]
    for i in range(int(n)):
        default_cond = default_conditions[i] if i<len(default_conditions) else None
        default_type = default_cond.get("type","simple").lower() if default_cond else "simple"
        cond_type = st.selectbox(f"{key_prefix} Condition {i+1} Type", ["Simple","Nested Group"], index=0 if default_type=="simple" else 1, key=f"{key_prefix}_condtype_{i}")
        if cond_type=="Simple":
            st.markdown(f"*Simple Condition {i+1}*")
            def_lt = default_cond.get("left_type","Indicator").title() if default_cond else "Indicator"
            left_type = st.selectbox("Left Type",["Indicator","Constant"], index=["Indicator","Constant"].index(def_lt), key=f"{key_prefix}_lefttype_{i}")
            if left_type=="Indicator":
                if indicator_list:
                    def_lval = default_cond.get("left_value", indicator_list[0]) if default_cond else indicator_list[0]
                    if def_lval not in indicator_list:
                        def_lval = indicator_list[0]
                    left_value = st.selectbox("Left Operand", indicator_list, index=indicator_list.index(def_lval), key=f"{key_prefix}_left_{i}")
                else:
                    left_value = ""
            else:
                def_lval = default_cond.get("left_value", 0) if default_cond else 0
                left_value = st.number_input("Left Constant", value=float(def_lval), format=fmt, key=f"{key_prefix}_leftconst_{i}")

            def_op = default_cond.get("operator",">") if default_cond else ">"
            operator = st.selectbox("Operator", [">","<",">=","<=","==","!="], index=[">","<",">=","<=","==","!="].index(def_op), key=f"{key_prefix}_op_{i}")

            def_rt = default_cond.get("right_type","Indicator").title() if default_cond else "Indicator"
            right_type= st.selectbox("Right Type", ["Indicator","Constant"], index=["Indicator","Constant"].index(def_rt), key=f"{key_prefix}_righttype_{i}")
            if right_type=="Indicator":
                if indicator_list:
                    def_rval = default_cond.get("right_value", indicator_list[0]) if default_cond else indicator_list[0]
                    if def_rval not in indicator_list:
                        def_rval = indicator_list[0]
                    right_value = st.selectbox("Right Operand", indicator_list, index=indicator_list.index(def_rval), key=f"{key_prefix}_right_{i}")
                else:
                    right_value = ""
            else:
                def_rval = default_cond.get("right_value", 0) if default_cond else 0
                right_value = st.number_input("Right Constant", value=float(def_rval), format=fmt, key=f"{key_prefix}_rightconst_{i}")

            conditions.append({
                "type":"simple",
                "left_type": left_type.lower(),
                "left_value": left_value,
                "operator": operator,
                "right_type": right_type.lower(),
                "right_value": right_value
            })
        else:
            st.markdown(f"*Nested Condition {i+1}*")
            nested_default = default_cond if (default_cond and default_cond.get("type","simple").lower()=="group") else None
            nested = get_condition_group(f"{key_prefix}_nested{i+1}", indicator_list, nested_default)
            conditions.append(nested)

    group = {"type":"group","logic":logic.upper(),"conditions":conditions}
    expr_generated = condition_to_expr(group)
    st.caption(f"Generated Expression: {expr_generated}")
    return group

# Defaults
default_long_start = {
    "logic":"AND","n":2,
    "conditions":[
        {"type":"simple","left_type":"Indicator","left_value":"SMA_5","operator":">","right_type":"Indicator","right_value":"SMA_10"},
        {"type":"simple","left_type":"Indicator","left_value":"Open","operator":">","right_type":"Indicator","right_value":"SMA_5"}
    ]
}
default_long_continue={
    "logic":"AND","n":1,
    "conditions":[
        {"type":"simple","left_type":"Indicator","left_value":"SMA_5","operator":">","right_type":"Indicator","right_value":"SMA_10"}
    ]
}
default_short_start={
    "logic":"AND","n":2,
    "conditions":[
        {"type":"simple","left_type":"Indicator","left_value":"SMA_5","operator":"<","right_type":"Indicator","right_value":"SMA_10"},
        {"type":"simple","left_type":"Indicator","left_value":"Open","operator":"<","right_type":"Indicator","right_value":"SMA_5"}
    ]
}
default_short_continue={
    "logic":"AND","n":1,
    "conditions":[
        {"type":"simple","left_type":"Indicator","left_value":"SMA_5","operator":"<","right_type":"Indicator","right_value":"SMA_10"}
    ]
}

# -----------------------------------------------------------------------------
# 3) Helper Functions
# -----------------------------------------------------------------------------
def compute_offset(entry_price, atr, method, value):
    if method=="absolute":
        return value
    elif method=="atr":
        return (atr* value) if pd.notna(atr) else 0
    elif method=="percentage":
        return entry_price*(value/100.0)
    else:
        return 0
def calculate_units(entered_currency, cost):
    if cost <= 0:
        raise ValueError("Cost must be a positive number")
    ratio = entered_currency / cost
    # If ratio is between 0 and 1, return 1; otherwise, return the floor of the ratio.
    return 1 if 0 < ratio < 1 else int(ratio)


def compute_entry_size_if_currency(entered_currency, row_open, stop_price, tick_size, tick_value):
    # If stop price is absent or effectively equal to the open price, return 1
    if pd.isna(stop_price) or abs(row_open - stop_price) < 1e-8:
        return 1  # Return integer 1 instead of 1.0
    dist = abs(row_open - stop_price)
    ticks = dist / tick_size
    cost = ticks * tick_value
    # If cost is zero, return 1 (ensuring an integer)
    if cost == 0:
        return 1
    # Use the calculate_units function which returns 1 if 0 < ratio < 1, otherwise the floor of the ratio.
    lots = calculate_units(entered_currency, cost)
    return lots  # This will be an integer

def compute_target_stop(entry_price, atr, side, Volatility_Zone, is_initial=True, reentry_index=0, params=None):
    if side=="Long":
        if Volatility_Zone=="High":
            if is_initial:
                t_m= params["init_target_method_long_high"]
                t_v= params["init_target_long_high"]
                s_m= params["init_stop_method_long_high"]
                s_v= params["init_stop_long_high"]
            else:
                t_m= params["reentry_target_method_long_high"][reentry_index]
                t_v= params["reentry_target_long_high"][reentry_index]
                s_m= params["reentry_stop_method_long_high"][reentry_index]
                s_v= params["reentry_stop_long_high"][reentry_index]
            targ_off= compute_offset(entry_price, atr, t_m, t_v) if params["target_defined"] else 0
            stop_off= compute_offset(entry_price, atr, s_m, s_v) if params["stop_defined"] else 0
            target_price= entry_price + targ_off if targ_off!=0 else np.nan
            stop_price=   entry_price - stop_off if stop_off!=0 else np.nan
        else:
            if is_initial:
                t_m= params["init_target_method_long_low"]
                t_v= params["init_target_long_low"]
                s_m= params["init_stop_method_long_low"]
                s_v= params["init_stop_long_low"]
            else:
                t_m= params["reentry_target_method_long_low"][reentry_index]
                t_v= params["reentry_target_long_low"][reentry_index]
                s_m= params["reentry_stop_method_long_low"][reentry_index]
                s_v= params["reentry_stop_long_low"][reentry_index]
            targ_off= compute_offset(entry_price, atr, t_m, t_v) if params["target_defined"] else 0
            stop_off= compute_offset(entry_price, atr, s_m, s_v) if params["stop_defined"] else 0
            target_price= entry_price + targ_off if targ_off!=0 else np.nan
            stop_price=   entry_price - stop_off if stop_off!=0 else np.nan
    else:
        if Volatility_Zone=="High":
            if is_initial:
                t_m= params["init_target_method_short_high"]
                t_v= params["init_target_short_high"]
                s_m= params["init_stop_method_short_high"]
                s_v= params["init_stop_short_high"]
            else:
                t_m= params["reentry_target_method_short_high"][reentry_index]
                t_v= params["reentry_target_short_high"][reentry_index]
                s_m= params["reentry_stop_method_short_high"][reentry_index]
                s_v= params["reentry_stop_short_high"][reentry_index]
            targ_off= compute_offset(entry_price, atr, t_m, t_v) if params["target_defined"] else 0
            stop_off= compute_offset(entry_price, atr, s_m, s_v) if params["stop_defined"] else 0
            target_price= entry_price - targ_off if targ_off!=0 else np.nan
            stop_price=   entry_price + stop_off if stop_off!=0 else np.nan
        else:
            if is_initial:
                t_m= params["init_target_method_short_low"]
                t_v= params["init_target_short_low"]
                s_m= params["init_stop_method_short_low"]
                s_v= params["init_stop_short_low"]
            else:
                t_m= params["reentry_target_method_short_low"][reentry_index]
                t_v= params["reentry_target_short_low"][reentry_index]
                s_m= params["reentry_stop_method_short_low"][reentry_index]
                s_v= params["reentry_stop_short_low"][reentry_index]
            targ_off= compute_offset(entry_price, atr, t_m, t_v) if params["target_defined"] else 0
            stop_off= compute_offset(entry_price, atr, s_m, s_v) if params["stop_defined"] else 0
            target_price= entry_price - targ_off if targ_off!=0 else np.nan
            stop_price=   entry_price + stop_off if stop_off!=0 else np.nan

    return {"target_price": target_price, "stop_price": stop_price}

# -----------------------------------------------------------------------------
# 4) apply_trade_setup_df - Normal Simulation
# -----------------------------------------------------------------------------
def apply_trade_setup_df(df, params):
    """
    Full normal simulation logic, storing param fields in both local e_dict 
    and the open event 'New Trend Entry'/'Reentry' for easy retrieval in get_normalized_trades.
    """
    if df.empty:
        return df, []

    df = df.reset_index(drop=True)
    trade_events=[]

    atr_col = "ATR_20"
    trend_id_counter=1

    # -------------------------------------------------------------------------
    # New Helper Functions for Per-Slot Counter Logic
    def update_slot_reuse(current_trend, slot):
        """
        Updates the slot reuse counter for the given slot in the current trend.
        Returns the updated usage count for that slot.
        """
        if "slot_reuse_counts" not in current_trend:
            current_trend["slot_reuse_counts"] = {}
        counts = current_trend["slot_reuse_counts"]
        if slot in counts:
            counts[slot] += 1
        else:
            counts[slot] = 1
        return counts[slot]

    def get_trade_label(slot, usage_count):
        """
        Generates a trade label based on the slot number and its usage count.
        For slot 0, the base label is 'Initial'; for other slots, 'Reentry n'.
        If the slot is reused (usage_count > 1), appends a suffix like '_1', '_2', etc.
        """
        if slot == 0:
            base_label = "Initial"
        else:
            base_label = f"Reentry {slot}"
        if usage_count == 1:
            return base_label
        else:
            return f"{base_label}_{usage_count - 1}"
    # -------------------------------------------------------------------------

    # Helper: check if user allows reentry after a STOP exit, etc.
    def stop_reentry_allowed(side, vol, is_init, re_idx):
        if side=="Long":
            if vol=="High":
                if is_init:
                    return params.get("init_stop_allow_reentry_long_high", False)
                else:
                    arr= params.get("reentry_stop_allow_reentry_long_high", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
            else:
                if is_init:
                    return params.get("init_stop_allow_reentry_long_low", False)
                else:
                    arr= params.get("reentry_stop_allow_reentry_long_low", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
        else:
            if vol=="High":
                if is_init:
                    return params.get("init_stop_allow_reentry_short_high", False)
                else:
                    arr= params.get("reentry_stop_allow_reentry_short_high", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
            else:
                if is_init:
                    return params.get("init_stop_allow_reentry_short_low", False)
                else:
                    arr= params.get("reentry_stop_allow_reentry_short_low", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False

    def target_reentry_allowed(side, vol, is_init, re_idx):
        if side=="Long":
            if vol=="High":
                if is_init:
                    return params.get("init_target_allow_reentry_long_high", False)
                else:
                    arr= params.get("reentry_target_allow_reentry_long_high", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
            else:
                if is_init:
                    return params.get("init_target_allow_reentry_long_low", False)
                else:
                    arr= params.get("reentry_target_allow_reentry_long_low", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
        else:
            if vol=="High":
                if is_init:
                    return params.get("init_target_allow_reentry_short_high", False)
                else:
                    arr= params.get("reentry_target_allow_reentry_short_high", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False
            else:
                if is_init:
                    return params.get("init_target_allow_reentry_short_low", False)
                else:
                    arr= params.get("reentry_target_allow_reentry_short_low", [])
                    return arr[re_idx] if (re_idx<len(arr)) else False

    def fill_entry_dict(e_dict, side, vol, is_init, re_idx, sm, sv, tm, tv):
        e_dict["Stop_Method"]   = sm
        e_dict["Stop_Value"]    = sv
        e_dict["Target_Method"] = tm
        e_dict["Target_Value"]  = tv

        def get_entry_ts():
            if side=="Long":
                if vol=="High":
                    if is_init:
                        return {
                            "type": params["init_entry_trailing_stop_type_long_high"],
                            "atr": params["init_entry_trailing_stop_atr_multiplier_long_high"],
                            "inds": params["init_entry_trailing_stop_indicator_long_high"],
                            "allow": params["init_entry_trailing_stop_allow_reentry_long_high"]
                        }
                    else:
                        return {
                            "type": params["reentry_entry_trailing_stop_type_long_high"][re_idx],
                            "atr": params["reentry_entry_trailing_stop_atr_multiplier_long_high"][re_idx],
                            "inds": params["reentry_entry_trailing_stop_indicator_long_high"][re_idx],
                            "allow": params["reentry_entry_trailing_stop_allow_reentry_long_high"][re_idx]
                        }
                else:
                    if is_init:
                        return {
                            "type": params["init_entry_trailing_stop_type_long_low"],
                            "atr": params["init_entry_trailing_stop_atr_multiplier_long_low"],
                            "inds": params["init_entry_trailing_stop_indicator_long_low"],
                            "allow": params["init_entry_trailing_stop_allow_reentry_long_low"]
                        }
                    else:
                        return {
                            "type": params["reentry_entry_trailing_stop_type_long_low"][re_idx],
                            "atr": params["reentry_entry_trailing_stop_atr_multiplier_long_low"][re_idx],
                            "inds": params["reentry_entry_trailing_stop_indicator_long_low"][re_idx],
                            "allow": params["reentry_entry_trailing_stop_allow_reentry_long_low"][re_idx]
                        }
            else:
                if vol=="High":
                    if is_init:
                        return {
                            "type": params["init_entry_trailing_stop_type_short_high"],
                            "atr": params["init_entry_trailing_stop_atr_multiplier_short_high"],
                            "inds": params["init_entry_trailing_stop_indicator_short_high"],
                            "allow": params["init_entry_trailing_stop_allow_reentry_short_high"]
                        }
                    else:
                        return {
                            "type": params["reentry_entry_trailing_stop_type_short_high"][re_idx],
                            "atr": params["reentry_entry_trailing_stop_atr_multiplier_short_high"][re_idx],
                            "inds": params["reentry_entry_trailing_stop_indicator_short_high"][re_idx],
                            "allow": params["reentry_entry_trailing_stop_allow_reentry_short_high"][re_idx]
                        }
                else:
                    if is_init:
                        return {
                            "type": params["init_entry_trailing_stop_type_short_low"],
                            "atr": params["init_entry_trailing_stop_atr_multiplier_short_low"],
                            "inds": params["init_entry_trailing_stop_indicator_short_low"],
                            "allow": params["init_entry_trailing_stop_allow_reentry_short_low"]
                        }
                    else:
                        return {
                            "type": params["reentry_entry_trailing_stop_type_short_low"][re_idx],
                            "atr": params["reentry_entry_trailing_stop_atr_multiplier_short_low"][re_idx],
                            "inds": params["reentry_entry_trailing_stop_indicator_short_low"][re_idx],
                            "allow": params["reentry_entry_trailing_stop_allow_reentry_short_low"][re_idx]
                        }
        ts_= get_entry_ts()
        e_dict["entry_ts_type"]           = ts_["type"]
        e_dict["entry_ts_atr"]            = ts_["atr"]
        e_dict["entry_ts_inds"]           = ts_["inds"]
        e_dict["entry_ts_allow_reentry"]  = ts_["allow"]
        e_dict["entry_most_fav_price"]    = e_dict["entry_price"]

        def get_global_ts(side, vol):
            if side=="Long":
                if vol=="High":
                    return {
                        "type": params["global_trailing_stop_type_long_high"],
                        "atr": params["global_trailing_stop_atr_multiplier_long_high"],
                        "inds": params["global_trailing_stop_indicator_long_high"]
                    }
                else:
                    return {
                        "type": params["global_trailing_stop_type_long_low"],
                        "atr": params["global_trailing_stop_atr_multiplier_long_low"],
                        "inds": params["global_trailing_stop_indicator_long_low"]
                    }
            else:
                if vol=="High":
                    return {
                        "type": params["global_trailing_stop_type_short_high"],
                        "atr": params["global_trailing_stop_atr_multiplier_short_high"],
                        "inds": params["global_trailing_stop_indicator_short_high"]
                    }
                else:
                    return {
                        "type": params["global_trailing_stop_type_short_low"],
                        "atr": params["global_trailing_stop_atr_multiplier_short_low"],
                        "inds": params["global_trailing_stop_indicator_short_low"]
                    }
        glb= get_global_ts(side, vol)
        e_dict["global_ts_type"] = glb["type"]
        e_dict["global_ts_atr"]  = glb["atr"]
        e_dict["global_ts_inds"] = glb["inds"]

    def build_open_event(row_i, row_date, current_trend, slot_index, event_type, e_dict):
        """
        Creates the trade_events record for 'New Trend Entry' or 'Reentry' 
        that includes all the param fields so get_normalized_trades can read them directly.
        """
        return {
            "row": row_i,
            "date": row_date,
            "trend_id": current_trend["trend_id"],
            "entry_n": slot_index+1,
            "event": event_type,
            "Trade_Label": e_dict.get("Trade_Label", event_type),
            "entry_price": e_dict["entry_price"],
            "size": e_dict["size"],
            "direction": current_trend["direction"],
            "Volatility_Zone": e_dict["Volatility_Zone"],
            "Stop_Method": e_dict.get("Stop_Method"),
            "Stop_Value": e_dict.get("Stop_Value"),
            "Target_Method": e_dict.get("Target_Method"),
            "Target_Value": e_dict.get("Target_Value"),
            "Entry_TS_Type": e_dict.get("entry_ts_type"),
            "Entry_TS_ATR": e_dict.get("entry_ts_atr", 0.0),
            "Entry_TS_Indicators": e_dict.get("entry_ts_inds"),
            "entry_ts_allow_reentry": e_dict.get("entry_ts_allow_reentry"),
            "Global_TS_Type": e_dict.get("global_ts_type"),
            "Global_TS_ATR": e_dict.get("global_ts_atr", 0.0),
            "Global_TS_Indicators": e_dict.get("global_ts_inds"),
            "trend_most_fav_price": current_trend["most_fav_price"]
        }

    df_len= len(df)
    current_trend={
        "trend_id": None,
        "direction": 0,
        "start_price": np.nan,
        "most_fav_price": np.nan,
        "trend_active": False,
        "entries": [],
        "no_more_entries": False
    }

    for i in range(df_len):
        row_date = df.at[i,"Date"] if "Date" in df.columns else None
        row_open = df.at[i,"Open"]
        row_high = df.at[i,"High"]
        row_low  = df.at[i,"Low"]
        atr_coln = "ATR_20"
        row_atr  = df.at[i, atr_coln] if atr_coln in df.columns else np.nan
        vol_zone = str(df.at[i,"Volatility_Zone"]) if "Volatility_Zone" in df.columns else "Low"

        if i>0 and bool(df.at[i-1,"Trend_Active"]):
            current_trend["trend_id"]     = df.at[i-1,"Trend_ID"]
            current_trend["direction"]    = int(df.at[i-1,"Trend_Direction"])
            current_trend["start_price"]  = df.at[i-1,"Trend_Start_Price"]
            current_trend["most_fav_price"]=df.at[i-1,"Trend_Most_Favorable_Price"]
            current_trend["trend_active"] = True
            current_trend["no_more_entries"]= bool(df.at[i-1,"trend_no_more_entries"])
            tmp_ents=[]
            for en in range(1,7):
                is_open= df.at[i-1,f"Entry_{en}_IsOpen"]
                if is_open and not pd.isna(is_open):
                    e_dict={
                        "is_open": True,
                        "entry_price": df.at[i-1,f"Entry_{en}_Price"],
                        "size": df.at[i-1,f"Entry_{en}_Size"],
                        "stop_price": df.at[i-1,f"Entry_{en}_Stop"],
                        "target_price": df.at[i-1,f"Entry_{en}_Target"],
                        "exit_price": df.at[i-1,f"Entry_{en}_Exit_Price"],
                        "exit_method": df.at[i-1,f"Entry_{en}_Exit_Method"],
                        "entry_date": df.at[i-1,"Date"],
                        "Volatility_Zone": df.at[i-1,"Volatility_Zone"],
                        "slot_index": en-1,
                        "is_initial": (True if en==1 else False),
                        "reentry_index": (en-2 if en>1 else None),
                        "Stop_Method": None,
                        "Stop_Value": None,
                        "Target_Method": None,
                        "Target_Value": None,
                        "entry_ts_type": None,
                        "entry_ts_atr":  0.0,
                        "entry_ts_inds": [],
                        "entry_ts_allow_reentry": True,
                        "entry_most_fav_price": df.at[i-1,f"Entry_{en}_Price"],
                        "global_ts_type": None,
                        "global_ts_atr":  0.0,
                        "global_ts_inds": [],
                        "trend_most_fav_price": df.at[i-1,"Trend_Most_Favorable_Price"]
                    }
                    tmp_ents.append(e_dict)
            current_trend["entries"]= tmp_ents
        else:
            current_trend={
                "trend_id": None,
                "direction":0,
                "start_price": np.nan,
                "most_fav_price": np.nan,
                "trend_active": False,
                "entries": [],
                "no_more_entries":False
            }

        if current_trend["trend_active"]:
            if current_trend["direction"]==1:
                current_trend["most_fav_price"] = max(current_trend["most_fav_price"], row_open)
            else:
                current_trend["most_fav_price"] = min(current_trend["most_fav_price"], row_open)

        # global trailing stop?
        triggered_global=False
        if current_trend["trend_active"] and not current_trend["no_more_entries"]:
            side= "Long" if current_trend["direction"]==1 else "Short"
            reasons=[]
            def check_global_ts():
                if side=="Long":
                    if vol_zone=="High":
                        gtype= params["global_trailing_stop_type_long_high"]
                        gatr= params["global_trailing_stop_atr_multiplier_long_high"]
                        ginds= params["global_trailing_stop_indicator_long_high"]
                        retr= current_trend["most_fav_price"] - row_open
                        r=[]
                        if gtype in ("ATR","Both"):
                            if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>=gatr:
                                r.append("Global TS ATR")
                        if gtype in ("Indicator","Both"):
                            for indn in ginds:
                                val= df.loc[i, indn] if indn in df.columns else np.nan
                                if pd.notna(val) and row_open<= val:
                                    r.append(f"Global TS Indicator:{indn}")
                        return r
                    else:
                        gtype= params["global_trailing_stop_type_long_low"]
                        gatr= params["global_trailing_stop_atr_multiplier_long_low"]
                        ginds= params["global_trailing_stop_indicator_long_low"]
                        retr= current_trend["most_fav_price"] - row_open
                        r=[]
                        if gtype in ("ATR","Both"):
                            if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>=gatr:
                                r.append("Global TS ATR")
                        if gtype in ("Indicator","Both"):
                            for indn in ginds:
                                val= df.loc[i, indn] if indn in df.columns else np.nan
                                if pd.notna(val) and row_open<= val:
                                    r.append(f"Global TS Indicator:{indn}")
                        return r
                else:
                    if vol_zone=="High":
                        gtype= params["global_trailing_stop_type_short_high"]
                        gatr= params["global_trailing_stop_atr_multiplier_short_high"]
                        ginds= params["global_trailing_stop_indicator_short_high"]
                        retr= row_open- current_trend["most_fav_price"]
                        r=[]
                        if gtype in ("ATR","Both"):
                            if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>=gatr:
                                r.append("Global TS ATR")
                        if gtype in ("Indicator","Both"):
                            for indn in ginds:
                                val= df.loc[i, indn] if indn in df.columns else np.nan
                                if pd.notna(val) and row_open>= val:
                                    r.append(f"Global TS Indicator:{indn}")
                        return r
                    else:
                        gtype= params["global_trailing_stop_type_short_low"]
                        gatr= params["global_trailing_stop_atr_multiplier_short_low"]
                        ginds= params["global_trailing_stop_indicator_short_low"]
                        retr= row_open- current_trend["most_fav_price"]
                        r=[]
                        if gtype in ("ATR","Both"):
                            if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>=gatr:
                                r.append("Global TS ATR")
                        if gtype in ("Indicator","Both"):
                            for indn in ginds:
                                val= df.loc[i, indn] if indn in df.columns else np.nan
                                if pd.notna(val) and row_open>= val:
                                    r.append(f"Global TS Indicator:{indn}")
                        return r
            reasons= check_global_ts()
            if reasons:
                triggered_global=True
                for e_ in current_trend["entries"]:
                    if e_["is_open"]:
                        e_["is_open"]=False
                        e_["exit_price"]= row_open
                        e_["exit_method"]="Global Trailing Stop"
                        trade_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": e_["slot_index"]+1,
                            "event":"Global Trailing Stop",
                            "exit_date": row_date,
                            "exit_price": row_open
                        })
                current_trend["no_more_entries"]=True



        # check continue
        if current_trend["trend_active"]:
            side= "Long" if current_trend["direction"]==1 else "Short"
            if side=="Long":
                if vol_zone=="High":
                    cont_ = evaluate_condition_structured(params["long_high_continue"], df.loc[i])
                else:
                    cont_ = evaluate_condition_structured(params["long_low_continue"], df.loc[i])
            else:
                if vol_zone=="High":
                    cont_ = evaluate_condition_structured(params["short_high_continue"], df.loc[i])
                else:
                    cont_ = evaluate_condition_structured(params["short_low_continue"], df.loc[i])
            if not cont_:
                for e_ in current_trend["entries"]:
                    if e_["is_open"]:
                        e_["is_open"]=False
                        e_["exit_price"]= row_open
                        e_["exit_method"]="Trend Closure"
                        trade_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": e_["slot_index"]+1,
                            "event":"Trend Closure",
                            "exit_date": row_date,
                            "exit_price": row_open
                        })
                current_trend["trend_active"]= False

        # reentry
        if current_trend["trend_active"] and not current_trend["no_more_entries"]:
            side= "Long" if current_trend["direction"]==1 else "Short"
            if side=="Long":
                if vol_zone=="High":
                    reent= evaluate_condition_structured(params["long_high_reentry"], df.loc[i])
                else:
                    reent= evaluate_condition_structured(params["long_low_reentry"], df.loc[i])
            else:
                if vol_zone=="High":
                    reent= evaluate_condition_structured(params["short_high_reentry"], df.loc[i])
                else:
                    reent= evaluate_condition_structured(params["short_low_reentry"], df.loc[i])
            if reent:
                max_slots= 1 + params["max_reentries"]
                used=[]
                for e_ in current_trend["entries"]:
                    if e_["is_open"]:
                        used.append(e_["slot_index"])
                free=[]
                for s_ in range(max_slots):
                    occupant= [x for x in current_trend["entries"] if x["slot_index"]==s_]
                    if not occupant:
                        free.append(s_)
                    else:
                        ent= occupant[0]
                        if not ent["is_open"]:
                            exm= ent["exit_method"]
                            is_init_slot= ent["is_initial"]
                            re_idx_= ent["reentry_index"] if ent["reentry_index"] is not None else 0
                            allow_r= False
                            if exm=="Stop":
                                allow_r= stop_reentry_allowed(side, vol_zone, is_init_slot, re_idx_)
                            elif exm=="Target":
                                allow_r= target_reentry_allowed(side, vol_zone, is_init_slot, re_idx_)
                            elif exm=="Entry Trailing Stop":
                                allow_r= ent["entry_ts_allow_reentry"]
                            if allow_r:
                                free.append(s_)
                if free:
                    new_slot= free[0]
                    is_init_= (True if new_slot==0 else False)
                    re_idx_=  (new_slot-1 if new_slot>0 else 0)
                    # Update the per-slot counter and generate the trade label
                    usage_count = update_slot_reuse(current_trend, new_slot)
                    trade_label = get_trade_label(new_slot, usage_count)
                    from_ts= compute_target_stop(row_open, row_atr, side, vol_zone, is_init_, re_idx_, params)
                    if is_init_:
                        if side=="Long" and vol_zone=="High":
                            init_siz= params["init_long_high_size"]
                        elif side=="Long" and vol_zone=="Low":
                            init_siz= params["init_long_low_size"]
                        elif side=="Short" and vol_zone=="High":
                            init_siz= params["init_short_high_size"]
                        else:
                            init_siz= params["init_short_low_size"]
                    else:
                        if side=="Long" and vol_zone=="High":
                            init_siz= params["reentry_long_high_sizes"][re_idx_]
                        elif side=="Long" and vol_zone=="Low":
                            init_siz= params["reentry_long_low_sizes"][re_idx_]
                        elif side=="Short" and vol_zone=="High":
                            init_siz= params["reentry_short_high_sizes"][re_idx_]
                        else:
                            init_siz= params["reentry_short_low_sizes"][re_idx_]

                    if params["initial-entry-type"]=="currency":
                        sp_= from_ts["stop_price"]
                        actual_= compute_entry_size_if_currency(init_siz, row_open, sp_, params["tick_size"], params["tick_value"])
                    else:
                        actual_= init_siz

                    def get_stop_method_value():
                        if side=="Long":
                            if vol_zone=="High":
                                if is_init_:
                                    return (params["init_stop_method_long_high"], params["init_stop_long_high"])
                                else:
                                    return (params["reentry_stop_method_long_high"][re_idx_], params["reentry_stop_long_high"][re_idx_])
                            else:
                                if is_init_:
                                    return (params["init_stop_method_long_low"], params["init_stop_long_low"])
                                else:
                                    return (params["reentry_stop_method_long_low"][re_idx_], params["reentry_stop_long_low"][re_idx_])
                        else:
                            if vol_zone=="High":
                                if is_init_:
                                    return (params["init_stop_method_short_high"], params["init_stop_short_high"])
                                else:
                                    return (params["reentry_stop_method_short_high"][re_idx_], params["reentry_stop_short_high"][re_idx_])
                            else:
                                if is_init_:
                                    return (params["init_stop_method_short_low"], params["init_stop_short_low"])
                                else:
                                    return (params["reentry_stop_method_short_low"][re_idx_], params["reentry_stop_short_low"][re_idx_])
                    def get_target_method_value():
                        if side=="Long":
                            if vol_zone=="High":
                                if is_init_:
                                    return (params["init_target_method_long_high"], params["init_target_long_high"])
                                else:
                                    return (params["reentry_target_method_long_high"][re_idx_], params["reentry_target_long_high"][re_idx_])
                            else:
                                if is_init_:
                                    return (params["init_target_method_long_low"], params["init_target_long_low"])
                                else:
                                    return (params["reentry_target_method_long_low"][re_idx_], params["reentry_target_long_low"][re_idx_])
                        else:
                            if vol_zone=="High":
                                if is_init_:
                                    return (params["init_target_method_short_high"], params["init_target_short_high"])
                                else:
                                    return (params["reentry_target_method_short_high"][re_idx_], params["reentry_target_short_high"][re_idx_])
                            else:
                                if is_init_:
                                    return (params["init_target_method_short_low"], params["init_target_short_low"])
                                else:
                                    return (params["reentry_target_method_short_low"][re_idx_], params["reentry_target_short_low"][re_idx_])
                    sm_, sv_= get_stop_method_value()
                    tm_, tv_= get_target_method_value()

                    new_e={
                        "is_open": True,
                        "entry_price": row_open,
                        "size": actual_,
                        "stop_price": from_ts["stop_price"],
                        "target_price": from_ts["target_price"],
                        "exit_price": np.nan,
                        "exit_method": np.nan,
                        "entry_date": row_date,
                        "Volatility_Zone": vol_zone,
                        "slot_index": new_slot,
                        "is_initial": is_init_,
                        "reentry_index": (new_slot-1 if new_slot>0 else None),
                        "Trade_Label": trade_label
                    }
                    fill_entry_dict(new_e, side, vol_zone, is_init_, re_idx_, sm_, sv_, tm_, tv_)
                    new_e["trend_most_fav_price"]= current_trend["most_fav_price"]
                    current_trend["entries"].append(new_e)

                    open_evt= build_open_event(
                        row_i= i,
                        row_date= row_date,
                        current_trend= current_trend,
                        slot_index= new_slot,
                        event_type= "New Trend Entry" if is_init_ else "Reentry",
                        e_dict= new_e
                    )
                    trade_events.append(open_evt)





        # entry-level trailing
        if current_trend["trend_active"] and not triggered_global:
            side= "Long" if current_trend["direction"]==1 else "Short"
            for e_ in current_trend["entries"]:
                if e_["is_open"]:
                    if side=="Long":
                        e_["entry_most_fav_price"]= max(e_["entry_most_fav_price"], row_open)
                    else:
                        e_["entry_most_fav_price"]= min(e_["entry_most_fav_price"], row_open)
            for e_ in current_trend["entries"]:
                if e_["is_open"]:
                    tstype= e_["entry_ts_type"]
                    if tstype not in ("None","",None):
                        reasons=[]
                        if side=="Long":
                            retr= e_["entry_most_fav_price"]- row_open
                            if tstype in ("ATR","Both"):
                                if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>= e_["entry_ts_atr"]:
                                    reasons.append("Entry TS ATR")
                            if tstype in ("Indicator","Both"):
                                for indn in e_["entry_ts_inds"]:
                                    val= df.loc[i, indn] if indn in df.columns else np.nan
                                    if pd.notna(val) and row_open<= val:
                                        reasons.append(f"Entry TS Indicator:{indn}")
                        else:
                            retr= row_open- e_["entry_most_fav_price"]
                            if tstype in ("ATR","Both"):
                                if pd.notna(row_atr) and row_atr>0 and (retr/row_atr)>= e_["entry_ts_atr"]:
                                    reasons.append("Entry TS ATR")
                            if tstype in ("Indicator","Both"):
                                for indn in e_["entry_ts_inds"]:
                                    val= df.loc[i, indn] if indn in df.columns else np.nan
                                    if pd.notna(val) and row_open>= val:
                                        reasons.append(f"Entry TS Indicator:{indn}")
                        if reasons:
                            e_["is_open"]=False
                            e_["exit_price"]= row_open
                            e_["exit_method"]="Entry Trailing Stop"
                            trade_events.append({
                                "row": i,
                                "date": row_date,
                                "trend_id": current_trend["trend_id"],
                                "entry_n": e_["slot_index"]+1,
                                "event":"Entry Trailing Stop",
                                "exit_date": row_date,
                                "exit_price": row_open
                            })

        # normal stops/targets
        if current_trend["trend_active"]:
            side= "Long" if current_trend["direction"]==1 else "Short"
            for e_ in current_trend["entries"]:
                if e_["is_open"]:
                    stop_hit=False
                    target_hit=False
                    s_p= e_["stop_price"]
                    t_p= e_["target_price"]
                    if side=="Long":
                        if pd.notna(t_p) and row_high>= t_p:
                            target_hit=True
                        if pd.notna(s_p) and row_low<= s_p:
                            stop_hit=True
                    else:
                        if pd.notna(t_p) and row_low<= t_p:
                            target_hit=True
                        if pd.notna(s_p) and row_high>= s_p:
                            stop_hit=True
                    if stop_hit and target_hit:
                        if params["target_stop_preference"]=="stop":
                            target_hit=False
                        else:
                            stop_hit=False
                    if stop_hit:
                        e_["is_open"]=False
                        e_["exit_price"]= s_p
                        e_["exit_method"]="Stop"
                        trade_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": e_["slot_index"]+1,
                            "event":"Stop",
                            "exit_date": row_date,
                            "exit_price": s_p
                        })
                    elif target_hit:
                        e_["is_open"]=False
                        e_["exit_price"]= t_p
                        e_["exit_method"]="Target"
                        trade_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": e_["slot_index"]+1,
                            "event":"Target",
                            "exit_date": row_date,
                            "exit_price": t_p
                        })

        
        # if not active => new start
        if not current_trend["trend_active"]:
            side_candidate=None
            if vol_zone=="High":
                condL= evaluate_condition_structured(params["long_high_start"], df.loc[i])
                condS= evaluate_condition_structured(params["short_high_start"], df.loc[i])
                if params["preferred_high"]=="Long":
                    side_candidate= "Long" if condL else ("Short" if condS else None)
                else:
                    side_candidate= "Short" if condS else ("Long" if condL else None)
            else:
                condL= evaluate_condition_structured(params["long_low_start"], df.loc[i])
                condS= evaluate_condition_structured(params["short_low_start"], df.loc[i])
                if params["preferred_low"]=="Long":
                    side_candidate= "Long" if condL else ("Short" if condS else None)
                else:
                    side_candidate= "Short" if condS else ("Long" if condL else None)
            if side_candidate:
                current_trend["trend_id"]= trend_id_counter
                trend_id_counter+=1
                current_trend["direction"]= (1 if side_candidate=="Long" else -1)
                current_trend["start_price"]= row_open
                current_trend["most_fav_price"]= row_open
                current_trend["trend_active"]= True
                current_trend["no_more_entries"]= False
                current_trend["entries"]= []
                current_trend["slot_reuse_counts"] = {}  # Initialize the slot reuse counter
                from_ts= compute_target_stop(row_open, row_atr, side_candidate, vol_zone, is_initial=True, reentry_index=0, params=params)
                if side_candidate=="Long" and vol_zone=="High":
                    init_siz= params["init_long_high_size"]
                elif side_candidate=="Long" and vol_zone=="Low":
                    init_siz= params["init_long_low_size"]
                elif side_candidate=="Short" and vol_zone=="High":
                    init_siz= params["init_short_high_size"]
                else:
                    init_siz= params["init_short_low_size"]
                if params["initial-entry-type"]=="currency":
                    sp_ = from_ts["stop_price"]
                    actual_= compute_entry_size_if_currency(init_siz, row_open, sp_, params["tick_size"], params["tick_value"])
                else:
                    actual_= init_siz

                def stmv():
                    if side_candidate=="Long":
                        if vol_zone=="High":
                            return (params["init_stop_method_long_high"], params["init_stop_long_high"])
                        else:
                            return (params["init_stop_method_long_low"], params["init_stop_long_low"])
                    else:
                        if vol_zone=="High":
                            return (params["init_stop_method_short_high"], params["init_stop_short_high"])
                        else:
                            return (params["init_stop_method_short_low"], params["init_stop_short_low"])
                def ttmv():
                    if side_candidate=="Long":
                        if vol_zone=="High":
                            return (params["init_target_method_long_high"], params["init_target_long_high"])
                        else:
                            return (params["init_target_method_long_low"], params["init_target_long_low"])
                    else:
                        if vol_zone=="High":
                            return (params["init_target_method_short_high"], params["init_target_short_high"])
                        else:
                            return (params["init_target_method_short_low"], params["init_target_short_low"])
                sm_, sv_= stmv()
                tm_, tv_= ttmv()
                
                # Update counter and generate trade label for the initial slot (slot 0)
                usage_count = update_slot_reuse(current_trend, 0)
                trade_label = get_trade_label(0, usage_count)
                
                new_e={
                    "is_open": True,
                    "entry_price": row_open,
                    "size": actual_,
                    "stop_price": from_ts["stop_price"],
                    "target_price": from_ts["target_price"],
                    "exit_price": np.nan,
                    "exit_method": np.nan,
                    "entry_date": row_date,
                    "Volatility_Zone": vol_zone,
                    "slot_index": 0,
                    "is_initial": True,
                    "reentry_index": None,
                    "Trade_Label": trade_label
                }
                fill_entry_dict(new_e, side_candidate, vol_zone, True, 0, sm_, sv_, tm_, tv_)
                new_e["trend_most_fav_price"]= current_trend["most_fav_price"]
                current_trend["entries"].append(new_e)

                open_evt= build_open_event(
                    row_i= i,
                    row_date= row_date,
                    current_trend= current_trend,
                    slot_index= 0,
                    event_type= "New Trend Entry",
                    e_dict= new_e
                )
                trade_events.append(open_evt)

        if current_trend["trend_active"]:
            df.at[i,"Trend_Active"]= True
            df.at[i,"Trend_ID"]= current_trend["trend_id"]
            df.at[i,"Trend_Direction"]= current_trend["direction"]
            df.at[i,"Trend_Start_Price"]= current_trend["start_price"]
            df.at[i,"Trend_Most_Favorable_Price"]= current_trend["most_fav_price"]
            d_ = current_trend["direction"]
            if pd.notna(row_open) and pd.notna(current_trend["start_price"]):
                df.at[i,"Trend_Profit_Price"]= (row_open- current_trend["start_price"])* d_
                if pd.notna(row_atr) and row_atr!=0:
                    df.at[i,"Trend_Profit_ATR"]= df.at[i,"Trend_Profit_Price"]/ row_atr
                else:
                    df.at[i,"Trend_Profit_ATR"]= np.nan
            else:
                df.at[i,"Trend_Profit_Price"]= np.nan
                df.at[i,"Trend_Profit_ATR"]= np.nan

            df.at[i,"trend_no_more_entries"]= current_trend["no_more_entries"]
            df.at[i,"Open_Entries_Count"]= sum(e["is_open"] for e in current_trend["entries"])
            df.at[i,"Reentry_Count"]= sum(1 for e in current_trend["entries"] if not e["is_initial"])
        else:
            df.at[i,"Trend_Active"]= False
            df.at[i,"Trend_ID"]= np.nan
            df.at[i,"Trend_Direction"]= np.nan
            df.at[i,"Trend_Start_Price"]= np.nan
            df.at[i,"Trend_Most_Favorable_Price"]= np.nan
            df.at[i,"Trend_Profit_Price"]= np.nan
            df.at[i,"Trend_Profit_ATR"]= np.nan
            df.at[i,"trend_no_more_entries"]= False
            df.at[i,"Open_Entries_Count"]= 0
            df.at[i,"Reentry_Count"]= 0

        for en in range(1,7):
            df.at[i, f"Entry_{en}_IsOpen"]= False
            df.at[i, f"Entry_{en}_Price"]= np.nan
            df.at[i, f"Entry_{en}_Size"]= np.nan
            df.at[i, f"Entry_{en}_Stop"]= np.nan
            df.at[i, f"Entry_{en}_Target"]= np.nan
            df.at[i, f"Entry_{en}_Exit_Price"]= np.nan
            df.at[i, f"Entry_{en}_Exit_Method"]= np.nan

        if current_trend["trend_active"]:
            for e_ in current_trend["entries"]:
                n_ = e_["slot_index"]+1
                if n_<=6:
                    df.at[i,f"Entry_{n_}_IsOpen"]= e_["is_open"]
                    df.at[i,f"Entry_{n_}_Price"]= e_["entry_price"]
                    df.at[i,f"Entry_{n_}_Size"]= e_["size"]
                    df.at[i,f"Entry_{n_}_Stop"]= e_["stop_price"]
                    df.at[i,f"Entry_{n_}_Target"]= e_["target_price"]
                    df.at[i,f"Entry_{n_}_Exit_Price"]= e_["exit_price"]
                    df.at[i,f"Entry_{n_}_Exit_Method"]= e_["exit_method"]

    return df, trade_events

# -----------------------------------------------------------------------------
# 5) simulate_variant
# -----------------------------------------------------------------------------
def simulate_variant(df, params, stop_multiplier, target_multiplier):
    if df.empty:
        return df, []

    df = df.reset_index(drop=True)
    variant_events=[]
    atr_col= "ATR_20"
    trend_id_counter=1
    current_trend={
        "trend_id":None,
        "direction":0,
        "start_price":np.nan,
        "most_fav_price":np.nan,
        "trend_active":False,
        "entries":[],
        "no_more_entries":False
    }

    def compute_variant_stop_target(open_price, row_atr, side):
        so= row_atr* stop_multiplier if pd.notna(row_atr) else 0
        if side=="Long":
            stop_p= open_price- so
            target_p= open_price+ so* target_multiplier
        else:
            stop_p= open_price+ so
            target_p= open_price- so* target_multiplier
        return stop_p, target_p

    nrows= len(df)
    for i in range(nrows):
        row_date= df.at[i,"Date"] if "Date" in df.columns else None
        row_open= df.at[i,"Open"]
        row_high= df.at[i,"High"]
        row_low= df.at[i,"Low"]
        row_atr= df.at[i,atr_col] if atr_col in df.columns else np.nan
        vol_zone= str(df.at[i,"Volatility_Zone"]) if "Volatility_Zone" in df.columns else "Low"

        if i>0 and bool(df.at[i-1,"Trend_Active"]):
            current_trend["trend_id"]= df.at[i-1,"Trend_ID"]
            current_trend["direction"]= int(df.at[i-1,"Trend_Direction"])
            current_trend["start_price"]= df.at[i-1,"Trend_Start_Price"]
            current_trend["most_fav_price"]= df.at[i-1,"Trend_Most_Favorable_Price"]
            current_trend["trend_active"]= True
            current_trend["no_more_entries"]= bool(df.at[i-1,"trend_no_more_entries"])
            tmp_ents=[]
            for en in range(1,7):
                is_open= df.at[i-1,f"Entry_{en}_IsOpen"]
                if is_open and not pd.isna(is_open):
                    e_d={
                        "is_open": True,
                        "entry_price": df.at[i-1,f"Entry_{en}_Price"],
                        "size": df.at[i-1,f"Entry_{en}_Size"],
                        "stop_price": df.at[i-1,f"Entry_{en}_Stop"],
                        "target_price": df.at[i-1,f"Entry_{en}_Target"],
                        "exit_price": df.at[i-1,f"Entry_{en}_Exit_Price"],
                        "exit_method": df.at[i-1,f"Entry_{en}_Exit_Method"],
                        "entry_date": df.at[i-1,"Date"],
                        "Volatility_Zone": df.at[i-1,"Volatility_Zone"]
                    }
                    tmp_ents.append(e_d)
            current_trend["entries"]= tmp_ents
        else:
            current_trend={
                "trend_id":None,
                "direction":0,
                "start_price":np.nan,
                "most_fav_price":np.nan,
                "trend_active":False,
                "entries":[],
                "no_more_entries":False
            }

        if current_trend["trend_active"]:
            if current_trend["direction"]==1:
                current_trend["most_fav_price"]= max(current_trend["most_fav_price"], row_high)
            else:
                current_trend["most_fav_price"]= min(current_trend["most_fav_price"], row_low)

        if current_trend["trend_active"]:
            side= "Long" if current_trend["direction"]==1 else "Short"
            for idx,e_ in enumerate(current_trend["entries"]):
                if e_["is_open"]:
                    stopP, targP = compute_variant_stop_target(e_["entry_price"], row_atr, side)
                    stop_hit=False
                    targ_hit=False
                    if side=="Long":
                        if pd.notna(targP) and row_high>= targP:
                            targ_hit=True
                        if pd.notna(stopP) and row_low<= stopP:
                            stop_hit=True
                    else:
                        if pd.notna(targP) and row_low<= targP:
                            targ_hit=True
                        if pd.notna(stopP) and row_high>= stopP:
                            stop_hit=True
                    if stop_hit and targ_hit:
                        if params.get("target_stop_preference","stop")=="stop":
                            targ_hit=False
                        else:
                            stop_hit=False
                    if stop_hit or targ_hit:
                        e_["is_open"]= False
                        e_["exit_price"]= stopP if stop_hit else targP
                        e_["exit_method"]= "Stop" if stop_hit else "Target"
                        variant_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": idx+1,
                            "event": e_["exit_method"],
                            "exit_date": row_date,
                            "exit_price": e_["exit_price"]
                        })
            if side=="Long":
                if vol_zone=="High":
                    cont_= evaluate_condition_structured(params["long_high_continue"], df.loc[i])
                else:
                    cont_= evaluate_condition_structured(params["long_low_continue"], df.loc[i])
            else:
                if vol_zone=="High":
                    cont_= evaluate_condition_structured(params["short_high_continue"], df.loc[i])
                else:
                    cont_= evaluate_condition_structured(params["short_low_continue"], df.loc[i])
            if not cont_:
                for idx,e_ in enumerate(current_trend["entries"]):
                    if e_["is_open"]:
                        e_["is_open"]= False
                        e_["exit_price"]= row_open
                        e_["exit_method"]= "Trend Closure"
                        variant_events.append({
                            "row": i,
                            "date": row_date,
                            "trend_id": current_trend["trend_id"],
                            "entry_n": idx+1,
                            "event": "Trend Closure",
                            "exit_date": row_date,
                            "exit_price": row_open
                        })
                current_trend["trend_active"]= False

        if not current_trend["trend_active"]:
            side_c=None
            if vol_zone=="High":
                cl= evaluate_condition_structured(params["long_high_start"], df.loc[i])
                cs= evaluate_condition_structured(params["short_high_start"], df.loc[i])
                if params.get("preferred_high","Long")=="Long":
                    side_c= "Long" if cl else ("Short" if cs else None)
                else:
                    side_c= "Short" if cs else ("Long" if cl else None)
            else:
                cl= evaluate_condition_structured(params["long_low_start"], df.loc[i])
                cs= evaluate_condition_structured(params["short_low_start"], df.loc[i])
                if params.get("preferred_low","Long")=="Long":
                    side_c= "Long" if cl else ("Short" if cs else None)
                else:
                    side_c= "Short" if cs else ("Long" if cl else None)
            if side_c:
                current_trend["trend_id"]= trend_id_counter
                trend_id_counter+=1
                current_trend["direction"]= (1 if side_c=="Long" else -1)
                current_trend["start_price"]= row_open
                current_trend["most_fav_price"]= row_open
                current_trend["trend_active"]=True
                current_trend["entries"]=[]
                s_p,t_p = compute_variant_stop_target(row_open, row_atr, side_c)
                new_e={
                    "is_open":True,
                    "entry_price": row_open,
                    "size":1.0,
                    "stop_price": s_p,
                    "target_price": t_p,
                    "exit_price": np.nan,
                    "exit_method": np.nan,
                    "entry_date": row_date,
                    "Volatility_Zone": vol_zone
                }
                current_trend["entries"].append(new_e)
                variant_events.append({
                    "row": i,
                    "date": row_date,
                    "trend_id": current_trend["trend_id"],
                    "entry_n":1,
                    "event":"New Trend Entry",
                    "entry_price": row_open,
                    "size":1.0,
                    "direction": current_trend["direction"],
                    "Volatility_Zone": vol_zone
                })

        if current_trend["trend_active"]:
            df.at[i,"Trend_Active"]= True
            df.at[i,"Trend_ID"]= current_trend["trend_id"]
            df.at[i,"Trend_Direction"]= current_trend["direction"]
        else:
            df.at[i,"Trend_Active"]= False
            df.at[i,"Trend_ID"]= np.nan
            df.at[i,"Trend_Direction"]= np.nan

    return df, variant_events

# -----------------------------------------------------------------------------
# 6) get_normalized_trades
# -----------------------------------------------------------------------------
def get_normalized_trades(trade_events_list, tick_size, tick_value):
    """
    Final trade list, merging the param fields from the open event 
    with the exit event.
    Because we improved the code: each 'New Trend Entry' or 'Reentry'
    event in trade_events now contains those param fields, plus the new Trade_Label.
    """
    normalized_trades=[]
    open_events={}  
    sorted_events= sorted(trade_events_list, key=lambda x: x["row"])

    for ev in sorted_events:
        tid= ev.get("trend_id")
        e_n= ev.get("entry_n")
        ev_type= ev.get("event")
        key_= (tid, e_n)

        if ev_type in ["New Trend Entry","Reentry"]:
            open_events[key_]= ev

        elif ev_type in ["Stop","Target","Global Trailing Stop","Entry Trailing Stop","Trend Closure"]:
            if key_ in open_events:
                open_evt= open_events[key_]
                direction= open_evt.get("direction",1)
                eprice= open_evt.get("entry_price", None)
                xprice= ev.get("exit_price", None)
                size_ = open_evt.get("size", 1.0)
                exit_date= ev.get("date", None)
                entry_date= open_evt.get("date",None)
                if eprice is None or xprice is None:
                    continue
                pnl_price= (xprice- eprice)* size_ if direction==1 else (eprice- xprice)* size_
                ticks= pnl_price / tick_size
                pnl_curr= ticks* tick_value

                rowdict={
                    "Entry_Date": entry_date,
                    "Entry_Price": eprice,
                    "Direction": direction,
                    "Volatility_Zone": open_evt.get("Volatility_Zone"),
                    "Trend_ID": tid,
                    "Entry_No": e_n,
                    "Exit_Price": xprice,
                    "Exit_Method": ev_type,
                    "Exit_Date": exit_date,
                    "PnL_Price": pnl_price,
                    "PnL_Currency": pnl_curr,
                    "Entry_Size": size_
                }
                rowdict["Stop_Method"]   = open_evt.get("Stop_Method", None)
                rowdict["Stop_Value"]    = open_evt.get("Stop_Value", None)
                rowdict["Target_Method"] = open_evt.get("Target_Method", None)
                rowdict["Target_Value"]  = open_evt.get("Target_Value", None)
                rowdict["Entry_TS_Type"] = open_evt.get("Entry_TS_Type", None)
                rowdict["Entry_TS_ATR"]  = open_evt.get("Entry_TS_ATR", 0.0)
                inds_ = open_evt.get("Entry_TS_Indicators", [])
                if isinstance(inds_, list):
                    rowdict["Entry_TS_Indicators"]= ",".join(inds_)
                else:
                    rowdict["Entry_TS_Indicators"]= str(inds_)
                rowdict["Entry_TS_MostFavPrice"]= open_evt.get("entry_most_fav_price", None)  
                rowdict["Global_TS_Type"]   = open_evt.get("Global_TS_Type", None)
                rowdict["Global_TS_ATR"]    = open_evt.get("Global_TS_ATR", 0.0)
                ginds_ = open_evt.get("Global_TS_Indicators", [])
                if isinstance(ginds_, list):
                    rowdict["Global_TS_Indicators"]= ",".join(ginds_)
                else:
                    rowdict["Global_TS_Indicators"]= str(ginds_)
                rowdict["Trend_TS_MostFavPrice"]= open_evt.get("trend_most_fav_price", None)
                rowdict["Trade_Label"] = open_evt.get("Trade_Label", "")
                normalized_trades.append(rowdict)
                del open_events[key_]

    return pd.DataFrame(normalized_trades)

def simulate_ma_combo(df, sim_params):
    """
    Simulate trades using a single MA combination condition (from sim_params["manual_combos"])
    following the same logic and ordering as apply_trade_setup_df.
    
    sim_params should include:
      - "manual_combos": list containing one MA combination, e.g. [["SMA_5", "SMA_20"]].
         (The combination is assumed to be sorted by period in ascending order so that the fastest MA is first.)
      - "preferred_high": Preferred side for High volatility ("Long" or "Short").
      - "preferred_low": Preferred side for Low volatility ("Long" or "Short").
      - "fixed_lots": Dictionary with keys: "Long_High", "Long_Low", "Short_High", "Short_Low".
      - "tick_size": Tick size (a number) for profit calculations.
      - "tick_value": Tick value (a number) for profit calculations.
      - "output_type": "trade_list", "final_df", or "both".
      
    Behavior:
      - For initial entry:
          * For Long: Checks that the MA order condition holds (i.e. fastest MA > next MA(s))
            AND that the Open price is greater than the fastest MA's value.
          * For Short: Checks that the MA order condition holds (i.e. fastest MA < next MA(s))
            AND that the Open price is less than the fastest MA's value.
      - For continuation: Only the MA order condition is checked.
      - If a trend is closed in a row, the function immediately re‑checks the entry conditions in the same row,
        first using the preferred side (based on sim_params["preferred_high"] or sim_params["preferred_low"])
        and then checking the alternate side.
      - The function uses the "Volatility_Zone" column in df to determine the variant and to pick the correct fixed lot.
      
    The normalized trade list is recorded with the following column names (matching trade_setup):
        "Entry_Date", "Entry_Price", "Direction", "Volatility_Zone", "Trend_ID", "Entry_No",
        "Exit_Price", "Exit_Method", "Exit_Date", "PnL_Price", "PnL_Currency", "Entry_Size", "Trade_Label"
        
    Returns:
      A tuple: (final_df, normalized_trade_df), where one may be None depending on sim_params["output_type"].
    """
    # Use the provided single combination.
    if not sim_params.get("manual_combos"):
        return df, []
    combo = sim_params["manual_combos"][0]  # e.g. ["SMA_5", "SMA_20"]

    nrows = len(df)
    df_sim = df.copy()
    # Initialize columns for trend tracking.
    df_sim["Trend_Active"] = False
    df_sim["Trend_ID"] = np.nan
    df_sim["Entry_Price"] = np.nan
    df_sim["Entry_Date"] = None
    df_sim["Exit_Price"] = np.nan
    df_sim["Exit_Date"] = None

    active_trend = None  # Will store current trend details.
    trend_id_counter = 1
    trade_events = []

    # Helper functions for MA condition checks.
    def check_ma_order_long(row, combo):
        try:
            values = [row[ma] for ma in combo]
        except Exception:
            return False
        # For Long, require descending order: fastest > slower(s)
        return all(earlier > later for earlier, later in zip(values, values[1:]))

    def check_ma_order_short(row, combo):
        try:
            values = [row[ma] for ma in combo]
        except Exception:
            return False
        # For Short, require ascending order: fastest < slower(s)
        return all(earlier < later for earlier, later in zip(values, values[1:]))

    def check_open_long(row, combo):
        try:
            fastest = combo[0]
        except Exception:
            return False
        return row["Open"] > row[fastest]

    def check_open_short(row, combo):
        try:
            fastest = combo[0]
        except Exception:
            return False
        return row["Open"] < row[fastest]

    # Process each row. We'll allow rechecking within the same row.
    for i in range(nrows):
        row = df_sim.loc[i]
        vol_zone = row["Volatility_Zone"] if "Volatility_Zone" in row else "Low"

        recheck = True
        while recheck:
            if active_trend is None:
                # No active trend; check both Long and Short initial conditions.
                cond_long = check_ma_order_long(row, combo) and check_open_long(row, combo)
                cond_short = check_ma_order_short(row, combo) and check_open_short(row, combo)
                
                side_candidate = None
                if vol_zone == "High":
                    if sim_params.get("preferred_high", "Long") == "Long":
                        side_candidate = "Long" if cond_long else ("Short" if cond_short else None)
                    else:
                        side_candidate = "Short" if cond_short else ("Long" if cond_long else None)
                else:
                    if sim_params.get("preferred_low", "Long") == "Long":
                        side_candidate = "Long" if cond_long else ("Short" if cond_short else None)
                    else:
                        side_candidate = "Short" if cond_short else ("Long" if cond_long else None)
                
                if side_candidate is not None:
                    active_trend = {
                        "trend_id": trend_id_counter,
                        "side": side_candidate,  # "Long" or "Short"
                        "entry_price": row["Open"],
                        "entry_date": row["Date"],
                        "vol_zone": vol_zone,
                        "combo": combo
                    }
                    trend_id_counter += 1
                    # Determine fixed lot size.
                    if side_candidate == "Long":
                        lot = sim_params["fixed_lots"]["Long_High"] if vol_zone=="High" else sim_params["fixed_lots"]["Long_Low"]
                    else:
                        lot = sim_params["fixed_lots"]["Short_High"] if vol_zone=="High" else sim_params["fixed_lots"]["Short_Low"]
                    trade_events.append({
                        "row": i,
                        "date": row["Date"],
                        "trend_id": active_trend["trend_id"],
                        "event": "New Trend Entry",
                        "entry_price": row["Open"],
                        "side": side_candidate,
                        "vol_zone": vol_zone,
                        "combo": combo,
                        "fixed_size": lot,
                        "Trade_Label": "Initial"  # For consistency.
                    })
                    df_sim.at[i, "Trend_Active"] = True
                    df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                    df_sim.at[i, "Entry_Price"] = row["Open"]
                    df_sim.at[i, "Entry_Date"] = row["Date"]
                    recheck = False  # New trend started; move to next row.
                else:
                    recheck = False  # No trend started; exit loop.
            else:
                # An active trend exists; check continuation condition.
                if active_trend["side"] == "Long":
                    cont = check_ma_order_long(row, combo)
                else:
                    cont = check_ma_order_short(row, combo)
                if not cont:
                    # Close the active trend.
                    trade_events.append({
                        "row": i,
                        "date": row["Date"],
                        "trend_id": active_trend["trend_id"],
                        "event": "Trend Closure",
                        "exit_price": row["Open"],
                        "side": active_trend["side"],
                        "vol_zone": active_trend["vol_zone"],
                        "combo": combo,
                        "Exit_Method": "Trend Closure"  # Explicitly record the method.
                    })
                    df_sim.at[i, "Trend_Active"] = False
                    df_sim.at[i, "Exit_Price"] = row["Open"]
                    df_sim.at[i, "Exit_Date"] = row["Date"]
                    # Reset active trend and re-check the entry conditions within the same row.
                    active_trend = None
                    recheck = True
                else:
                    # Trend continues.
                    df_sim.at[i, "Trend_Active"] = True
                    df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                    df_sim.at[i, "Entry_Price"] = active_trend["entry_price"]
                    df_sim.at[i, "Entry_Date"] = active_trend["entry_date"]
                    recheck = False

        # End of while loop for row i.
    
    # After processing all rows, if a trend is still active, close it using the last row's Open.
    if active_trend is not None:
        trade_events.append({
            "row": nrows-1,
            "date": df_sim.at[nrows-1, "Date"],
            "trend_id": active_trend["trend_id"],
            "event": "Trend Closure",
            "exit_price": df_sim.at[nrows-1, "Open"],
            "side": active_trend["side"],
            "vol_zone": active_trend["vol_zone"],
            "combo": combo,
            "Exit_Method": "Trend Closure"
        })
        df_sim.at[nrows-1, "Trend_Active"] = False
        df_sim.at[nrows-1, "Exit_Price"] = df_sim.at[nrows-1, "Open"]
        df_sim.at[nrows-1, "Exit_Date"] = df_sim.at[nrows-1, "Date"]
        active_trend = None

    # Normalize trade events into a trade list with the desired column names.
    normalized_trades = []
    # We'll assume each trade has a single entry (so Entry_No is always 1).
    for ev in trade_events:
        if ev["event"] == "Trend Closure":
            entry_ev = next((x for x in trade_events if x["trend_id"] == ev["trend_id"] and x["event"] == "New Trend Entry"), None)
            if entry_ev:
                eprice = entry_ev["entry_price"]
                xprice = ev["exit_price"]
                # For Long, direction = +1; for Short, direction = -1.
                direction = 1 if ev["side"] == "Long" else -1
                fixed_size = entry_ev["fixed_size"]
                pnl_price = (xprice - eprice) * direction * fixed_size
                ticks = pnl_price / sim_params["tick_size"] if sim_params["tick_size"] != 0 else np.nan
                pnl_curr = ticks * sim_params["tick_value"]
                normalized_trades.append({
                    "Entry_Date": entry_ev["date"],
                    "Entry_Price": eprice,
                    "Direction": direction,
                    "Volatility_Zone": ev["vol_zone"],
                    "Trend_ID": ev["trend_id"],
                    "Entry_No": 1,
                    "Exit_Price": xprice,
                    "Exit_Method": ev.get("Exit_Method", "Trend Closure"),
                    "Exit_Date": ev["date"],
                    "PnL_Price": pnl_price,
                    "PnL_Currency": pnl_curr,
                    "Entry_Size": fixed_size,
                    "Trade_Label": entry_ev.get("Trade_Label", "Initial")
                })
    
    out_type = sim_params.get("output_type", "both")
    if out_type == "trade_list":
        return None, pd.DataFrame(normalized_trades)
    elif out_type == "final_df":
        return df_sim, None
    else:
        return df_sim, pd.DataFrame(normalized_trades)
    
    
    
    
def simulate_indicator_percentile_combo(df, sim_params):
    """
    Simulate trades using a single MA combination condition together with an extra 
    indicator threshold condition (based on a selected percentile) for additional filtering.
    
    sim_params should include:
      - "manual_combos": list containing one MA combination, e.g. [["SMA_5", "SMA_20"]]
         (Assumed sorted in ascending order by period so that the fastest MA is first.)
      - "indicator": a string representing the selected indicator (e.g. "Volume").
      - "threshold": the percentile threshold (a number) for the selected indicator.
      - "direction": a string: ">" or "<", indicating if the condition is indicator > threshold or < threshold.
      - "include_open_condition": Boolean; if True, then the Open condition (Open > fastest MA for Long, or Open < fastest MA for Short) is added.
      - "preferred_high": Preferred side for High volatility ("Long" or "Short").
      - "preferred_low": Preferred side for Low volatility ("Long" or "Short").
      - "fixed_lots": Dictionary with keys "Long_High", "Long_Low", "Short_High", "Short_Low".
      - "tick_size": Tick size (number) for profit calculation.
      - "tick_value": Tick value (number) for profit calculation.
      - "output_type": "trade_list", "final_df", or "both".
      
    Processing:
      - Initial entry condition for Long:
           * MA condition: fastest MA > slower MA(s)
           * Indicator condition: row[indicator] > threshold if direction is ">", else row[indicator] < threshold (for Short, MA condition is reversed)
           * If include_open_condition is True: also require Open > row[fastest MA] for Long (and Open < row[fastest MA] for Short)
      - For Short, the MA condition is fastest MA < slower MA(s) and similar indicator condition.
      - Continuation condition: Only the MA condition.
      - The simulation uses the "Volatility_Zone" column to determine which fixed lot to apply.
      - If an active trend is closed in a row, the function re-checks the entry conditions in the same row.
      
    Returns:
      A tuple: (final_df, normalized_trade_df) where the normalized trade list DataFrame has columns:
          "Entry_Date", "Entry_Price", "Direction", "Volatility_Zone", "Trend_ID", "Entry_No",
          "Exit_Price", "Exit_Method", "Exit_Date", "PnL_Price", "PnL_Currency", "Entry_Size", "Trade_Label"
    """
    # Use provided MA combination.
    if not sim_params.get("manual_combos"):
        return df, []
    combo = sim_params["manual_combos"][0]  # e.g. ["SMA_5", "SMA_20"]

    nrows = len(df)
    df_sim = df.copy()
    df_sim["Trend_Active"] = False
    df_sim["Trend_ID"] = np.nan
    df_sim["Entry_Price"] = np.nan
    df_sim["Entry_Date"] = None
    df_sim["Exit_Price"] = np.nan
    df_sim["Exit_Date"] = None

    active_trend = None
    trend_id_counter = 1
    trade_events = []

    # Helper functions for MA checks.
    def check_ma_order_long(row, combo):
        try:
            values = [row[ma] for ma in combo]
        except Exception:
            return False
        return all(earlier > later for earlier, later in zip(values, values[1:]))

    def check_ma_order_short(row, combo):
        try:
            values = [row[ma] for ma in combo]
        except Exception:
            return False
        return all(earlier < later for earlier, later in zip(values, values[1:]))

    def check_open_long(row, combo):
        try:
            fastest = combo[0]
        except Exception:
            return False
        return row["Open"] > row[fastest]

    def check_open_short(row, combo):
        try:
            fastest = combo[0]
        except Exception:
            return False
        return row["Open"] < row[fastest]

    # Extra indicator condition.
    def check_indicator_condition(row, indicator, threshold, direction):
        try:
            val = row[indicator]
        except Exception:
            return False
        if direction == ">":
            return val > threshold
        else:
            return val < threshold

    # Process rows with a recheck loop to allow new trend entry within the same row.
    for i in range(nrows):
        row = df_sim.loc[i]
        vol_zone = row["Volatility_Zone"] if "Volatility_Zone" in row else "Low"

        recheck = True
        while recheck:
            if active_trend is None:
                # Evaluate initial conditions for both sides.
                # For Long:
                cond_ma_long = check_ma_order_long(row, combo)
                cond_ind_long = check_indicator_condition(row, sim_params["indicator"], sim_params["threshold"], ">")  if sim_params["direction"] == ">" else check_indicator_condition(row, sim_params["indicator"], sim_params["threshold"], "<")
                cond_open_long = check_open_long(row, combo) if sim_params.get("include_open_condition", False) else True
                cond_long = cond_ma_long and cond_ind_long and cond_open_long

                # For Short:
                cond_ma_short = check_ma_order_short(row, combo)
                cond_ind_short = check_indicator_condition(row, sim_params["indicator"], sim_params["threshold"], ">" ) if sim_params["direction"] == ">" else check_indicator_condition(row, sim_params["indicator"], sim_params["threshold"], "<")
                cond_open_short = check_open_short(row, combo) if sim_params.get("include_open_condition", False) else True
                cond_short = cond_ma_short and cond_ind_short and cond_open_short

                side_candidate = None
                if vol_zone == "High":
                    if sim_params.get("preferred_high", "Long") == "Long":
                        side_candidate = "Long" if cond_long else ("Short" if cond_short else None)
                    else:
                        side_candidate = "Short" if cond_short else ("Long" if cond_long else None)
                else:
                    if sim_params.get("preferred_low", "Long") == "Long":
                        side_candidate = "Long" if cond_long else ("Short" if cond_short else None)
                    else:
                        side_candidate = "Short" if cond_short else ("Long" if cond_long else None)

                if side_candidate is not None:
                    active_trend = {
                        "trend_id": trend_id_counter,
                        "side": side_candidate,
                        "entry_price": row["Open"],
                        "entry_date": row["Date"],
                        "vol_zone": vol_zone,
                        "combo": combo
                    }
                    trend_id_counter += 1
                    # Determine fixed lot size.
                    if side_candidate == "Long":
                        lot = sim_params["fixed_lots"]["Long_High"] if vol_zone=="High" else sim_params["fixed_lots"]["Long_Low"]
                    else:
                        lot = sim_params["fixed_lots"]["Short_High"] if vol_zone=="High" else sim_params["fixed_lots"]["Short_Low"]
                    trade_events.append({
                        "row": i,
                        "date": row["Date"],
                        "trend_id": active_trend["trend_id"],
                        "event": "New Trend Entry",
                        "entry_price": row["Open"],
                        "side": side_candidate,
                        "vol_zone": vol_zone,
                        "combo": combo,
                        "fixed_size": lot,
                        "Trade_Label": "Initial"
                    })
                    df_sim.at[i, "Trend_Active"] = True
                    df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                    df_sim.at[i, "Entry_Price"] = row["Open"]
                    df_sim.at[i, "Entry_Date"] = row["Date"]
                    recheck = False
                else:
                    recheck = False
            else:
                # Trend is active; check continuation condition (only MA condition).
                if active_trend["side"] == "Long":
                    cont = check_ma_order_long(row, combo)
                else:
                    cont = check_ma_order_short(row, combo)
                if not cont:
                    trade_events.append({
                        "row": i,
                        "date": row["Date"],
                        "trend_id": active_trend["trend_id"],
                        "event": "Trend Closure",
                        "exit_price": row["Open"],
                        "side": active_trend["side"],
                        "vol_zone": active_trend["vol_zone"],
                        "combo": combo,
                        "Exit_Method": "Trend Closure"
                    })
                    df_sim.at[i, "Trend_Active"] = False
                    df_sim.at[i, "Exit_Price"] = row["Open"]
                    df_sim.at[i, "Exit_Date"] = row["Date"]
                    active_trend = None
                    recheck = True  # Re-check same row for new entry.
                else:
                    df_sim.at[i, "Trend_Active"] = True
                    df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                    df_sim.at[i, "Entry_Price"] = active_trend["entry_price"]
                    df_sim.at[i, "Entry_Date"] = active_trend["entry_date"]
                    recheck = False
        # End of recheck while loop.
    
    # If trend is still active at end, close it.
    if active_trend is not None:
        trade_events.append({
            "row": nrows-1,
            "date": df_sim.at[nrows-1, "Date"],
            "trend_id": active_trend["trend_id"],
            "event": "Trend Closure",
            "exit_price": df_sim.at[nrows-1, "Open"],
            "side": active_trend["side"],
            "vol_zone": active_trend["vol_zone"],
            "combo": combo,
            "Exit_Method": "Trend Closure"
        })
        df_sim.at[nrows-1, "Trend_Active"] = False
        df_sim.at[nrows-1, "Exit_Price"] = df_sim.at[nrows-1, "Open"]
        df_sim.at[nrows-1, "Exit_Date"] = df_sim.at[nrows-1, "Date"]
        active_trend = None

    # Normalize trade events to produce the trade list with required column names.
    normalized_trades = []
    for ev in trade_events:
        if ev["event"] == "Trend Closure":
            entry_ev = next((x for x in trade_events if x["trend_id"] == ev["trend_id"] and x["event"] == "New Trend Entry"), None)
            if entry_ev:
                eprice = entry_ev["entry_price"]
                xprice = ev["exit_price"]
                direction = 1 if ev["side"] == "Long" else -1
                fixed_size = entry_ev["fixed_size"]
                pnl_price = (xprice - eprice) * direction * fixed_size
                ticks = pnl_price / sim_params["tick_size"] if sim_params["tick_size"] != 0 else np.nan
                pnl_curr = ticks * sim_params["tick_value"]
                normalized_trades.append({
                    "Entry_Date": entry_ev["date"],
                    "Entry_Price": eprice,
                    "Direction": direction,
                    "Volatility_Zone": ev["vol_zone"],
                    "Trend_ID": ev["trend_id"],
                    "Entry_No": 1,
                    "Exit_Price": xprice,
                    "Exit_Method": ev.get("Exit_Method", "Trend Closure"),
                    "Exit_Date": ev["date"],
                    "PnL_Price": pnl_price,
                    "PnL_Currency": pnl_curr,
                    "Entry_Size": fixed_size,
                    "Trade_Label": entry_ev.get("Trade_Label", "Initial")
                })
    
    out_type = sim_params.get("output_type", "both")
    if out_type == "trade_list":
        return None, pd.DataFrame(normalized_trades)
    elif out_type == "final_df":
        return df_sim, None
    else:
        return df_sim, pd.DataFrame(normalized_trades)




import numpy as np
import pandas as pd
import json
from graphviz import Digraph
from fpdf import FPDF
import os

# --------------------------------------------
# Helper function: Build a Graphviz tree diagram from a condition tree.
# --------------------------------------------
def build_tree_graph(condition, graph=None, parent=None):
    """
    Recursively builds a Graphviz Digraph from a condition tree.
    Supports condition types:
      - "simple": For simple conditions.
      - "group": For nested groups (with a "logic" key, e.g., "AND" or "OR").
    For "percentile" type simple conditions, the label includes the percentile value.
    """
    if graph is None:
        graph = Digraph()
    node_id = str(id(condition))
    if condition["type"] == "simple":
        if condition["right_type"] == "percentile":
            node_label = f"{condition['left_value']} {condition['operator']} {condition['right_value']} (Perc {condition.get('percentile','')})"
        else:
            node_label = f"{condition['left_value']} {condition['operator']} {condition['right_value']}"
    elif condition["type"] == "group":
        node_label = f"Group ({condition.get('logic','AND')})"
    else:
        node_label = "Unknown"
    graph.node(node_id, node_label)
    if parent:
        graph.edge(parent, node_id)
    if condition["type"] == "group":
        for sub in condition.get("conditions", []):
            build_tree_graph(sub, graph, node_id)
    return graph

# --------------------------------------------
# Helper function: Generate PDF report with JSON and tree diagrams for each variant.
# --------------------------------------------
def save_condition_tree_pdf(variants_info, output_path):
    """
    Generates a PDF report that includes, for each simulation variant, its unique variant ID,
    the JSON representation of its condition trees, and the corresponding tree diagram images.
    
    variants_info: list of tuples (variant_id, variant_params) where variant_params is a dict
       containing keys: 'init_entry_condition_long', 'init_entry_condition_short',
                      'long_continue_condition', 'short_continue_condition'
    output_path: file path for the PDF report.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Temporary directory for tree images
    temp_dir = "./temp_tree_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    for variant_id, variant_params in variants_info:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Variant {variant_id}", ln=True)
        pdf.ln(4)
        
        pdf.set_font("Arial", "", 12)
        # For each condition tree, add JSON text and tree image.
        for key in ["init_entry_condition_long", "init_entry_condition_short", 
                    "long_continue_condition", "short_continue_condition"]:
            cond_tree = variant_params.get(key)
            pdf.multi_cell(0, 8, f"{key}:\n" + json.dumps(cond_tree, indent=2))
            pdf.ln(2)
            if cond_tree is not None:
                graph = build_tree_graph(cond_tree)
                image_filename = os.path.join(temp_dir, f"{variant_id}_{key}.png")
                graph.format = "png"
                graph.render(filename=image_filename, cleanup=True)
                # Calculate effective page width
                epw = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.cell(0, 8, f"Tree diagram for {key}:", ln=True)
                pdf.image(image_filename + ".png", w=epw)
                pdf.ln(4)
        pdf.ln(8)
    pdf.output(output_path)

# --------------------------------------------
# Final Simulation Function
# --------------------------------------------
def simulate_breakout_combo(df, sim_params):
    """
    Final simulation function that processes a single simulation variant based on unified condition trees
    for entry and continuation. Supported condition types:
      - "indicator": compare one column to another.
      - "constant": compare a column to a fixed value.
      - "percentile": compare a column’s value to a computed quantile from a chosen column.
    
    Additionally, if the Target/Stop (TS) variant is enabled, the TS levels are computed immediately at trade entry 
    (using the current ATR from "ATR_20" and the provided multipliers) and stored. Then on each subsequent row, 
    the function first checks if the stored TS levels are hit. If yes, the trade is exited (according to the chosen mode),
    otherwise the continuation condition is evaluated.
    
    sim_params must include:
      "init_entry_condition_long": unified condition tree (dict) for long entry.
      "init_entry_condition_short": unified condition tree (dict) for short entry.
      "long_continue_condition": unified condition tree (dict) for long continuation.
      "short_continue_condition": unified condition tree (dict) for short continuation.
         (If a continuation condition is not provided for a side, the corresponding entry condition is used.)
      
      -- Target/Stop Variant --
      "target_stop_variant": Boolean.
         If True, then:
           "target_multiplier": float,
           "stop_multiplier": float,
           "target_stop_mode": "immediate_exit" or "mark_exit_only",
           "target_stop_preference": "stop" or "target"
         (For Long: target = Entry + (ATR_20 * target_multiplier),
          stop = Entry - (ATR_20 * stop_multiplier);
          For Short: vice versa.)
      
      -- Other Settings --
      "preferred_high": "Long" or "Short" for high volatility.
      "preferred_low": "Long" or "Short" for low volatility.
      "fixed_lots": dict with keys "Long_High", "Long_Low", "Short_High", "Short_Low".
      "tick_size": number (for profit calculation).
      "tick_value": number.
      "output_type": "trade_list", "final_df", or "both".
    
    Processing steps:
      1. For each row, if no trade is active, evaluate both long and short entry conditions.
         If either returns True, a new trade (trend) is opened using the preferred side (if both are true).
         If TS is enabled, compute target and stop levels immediately and store them in active_trend.
      2. For every subsequent row while a trade is active:
         a. If TS is enabled, first check if the stored target or stop levels have been hit.
            - If hit, record an exit event (either immediately closing the trade or marking an exit per mode)
              and set active_trend to None.
         b. Then, evaluate the appropriate continuation condition:
            - For Long, use "long_continue_condition" (or fallback to "init_entry_condition_long").
            - For Short, use "short_continue_condition" (or fallback to "init_entry_condition_short").
         c. If the continuation condition fails, record a "Trend Closure" event and close the trade.
      3. At the end of the dataset, if a trade is still active, close it at the last row’s Open.
      4. Normalize all trade events into a final trade list with columns:
         "Entry_Date", "Entry_Price", "Direction", "Volatility_Zone", "Trend_ID", "Entry_No",
         "Exit_Price", "Exit_Method", "Exit_Date", "PnL_Price", "PnL_Currency", "Entry_Size",
         "Trade_Label", "Target_Price", "Stop_Price", "TS_Enabled".
    
    Returns:
      (final_df, normalized_trade_df)
    """
    nrows = len(df)
    df_sim = df.copy()
    df_sim["Trend_Active"] = False
    df_sim["Trend_ID"] = np.nan
    df_sim["Entry_Price"] = np.nan
    df_sim["Entry_Date"] = None
    df_sim["Exit_Price"] = np.nan
    df_sim["Exit_Date"] = None

    active_trend = None
    trend_id_counter = 1
    trade_events = []

    # Recursive evaluation of a condition tree.
    def evaluate_condition_structure(row, cond, full_df):
        if cond["type"] == "simple":
            left = row[cond["left_value"]] if cond["left_type"] == "indicator" else float(cond["left_value"])
            if cond["right_type"] == "indicator":
                right = row[cond["right_value"]]
            elif cond["right_type"] == "percentile":
                right = full_df[cond["right_value"]].quantile(float(cond["percentile"]) / 100)
            else:
                right = float(cond["right_value"])
            op = cond["operator"]
            return left > right if op == ">" else left < right
        elif cond["type"] == "group":
            results = [evaluate_condition_structure(row, sub, full_df) for sub in cond["conditions"]]
            return all(results) if cond.get("logic", "AND") == "AND" else any(results)
        else:
            return True

    def evaluate_unified_condition(row, condition_tree, full_df):
        if not condition_tree:
            return True
        return evaluate_condition_structure(row, condition_tree, full_df)

    def compute_target_stop(entry_price, atr_val, side, t_mult, s_mult):
        if side == "Long":
            return entry_price + atr_val * t_mult, entry_price - atr_val * s_mult
        else:
            return entry_price - atr_val * t_mult, entry_price + atr_val * s_mult

    # Retrieve condition trees.
    init_long = sim_params.get("init_entry_condition_long")
    init_short = sim_params.get("init_entry_condition_short")
    cont_long = sim_params.get("long_continue_condition", init_long)
    cont_short = sim_params.get("short_continue_condition", init_short)

    # Process each row.
    for i in range(nrows):
        row = df_sim.loc[i]
        vol_zone = row["Volatility_Zone"] if "Volatility_Zone" in row else "Low"

        # If no trade is active, evaluate entry conditions.
        if active_trend is None:
            long_valid = evaluate_unified_condition(row, init_long, df)
            short_valid = evaluate_unified_condition(row, init_short, df)
            if long_valid or short_valid:
                # If both valid, choose based on preferred side.
                if long_valid and short_valid:
                    side_candidate = sim_params.get("preferred_high", "Long") if vol_zone == "High" else sim_params.get("preferred_low", "Long")
                elif long_valid:
                    side_candidate = "Long"
                else:
                    side_candidate = "Short"
                active_trend = {
                    "trend_id": trend_id_counter,
                    "side": side_candidate,
                    "entry_price": row["Open"],
                    "entry_date": row["Date"],
                    "vol_zone": vol_zone
                }
                trend_id_counter += 1
                # Determine lot size.
                if side_candidate == "Long":
                    lot = sim_params["fixed_lots"]["Long_High"] if vol_zone == "High" else sim_params["fixed_lots"]["Long_Low"]
                else:
                    lot = sim_params["fixed_lots"]["Short_High"] if vol_zone == "High" else sim_params["fixed_lots"]["Short_Low"]
                # If TS is enabled, compute and store target and stop at entry.
                if sim_params.get("target_stop_variant"):
                    atr_val = row["ATR_20"] if "ATR_20" in row and pd.notna(row["ATR_20"]) else 0
                    target_p, stop_p = compute_target_stop(row["Open"], atr_val, side_candidate,
                                                            sim_params.get("target_multiplier"), sim_params.get("stop_multiplier"))
                    active_trend["target_price"] = target_p
                    active_trend["stop_price"] = stop_p
                    active_trend["TS_Enabled"] = True
                else:
                    active_trend["TS_Enabled"] = False
                trade_events.append({
                    "row": i,
                    "date": row["Date"],
                    "trend_id": active_trend["trend_id"],
                    "event": "New Trend Entry",
                    "entry_price": row["Open"],
                    "side": side_candidate,
                    "vol_zone": vol_zone,
                    "fixed_size": lot,
                    "Trade_Label": f"Initial_{active_trend['trend_id']}",
                    "Target_Price": active_trend.get("target_price"),
                    "Stop_Price": active_trend.get("stop_price"),
                    "TS_Enabled": active_trend.get("TS_Enabled")
                })
                df_sim.at[i, "Trend_Active"] = True
                df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                df_sim.at[i, "Entry_Price"] = row["Open"]
                df_sim.at[i, "Entry_Date"] = row["Date"]

        # If a trade is active, check exit conditions.
        if active_trend is not None:
            # If TS is enabled, check if target or stop (precomputed at entry) is hit.
            if active_trend.get("TS_Enabled"):
                target_p = active_trend.get("target_price")
                stop_p = active_trend.get("stop_price")
                hit_target = False
                hit_stop = False
                if active_trend["side"] == "Long":
                    if row["High"] >= target_p:
                        hit_target = True
                    if row["Low"] <= stop_p:
                        hit_stop = True
                else:
                    if row["Low"] <= target_p:
                        hit_target = True
                    if row["High"] >= stop_p:
                        hit_stop = True
                if hit_stop and hit_target:
                    if sim_params.get("target_stop_preference", "stop") == "stop":
                        hit_target = False
                    else:
                        hit_stop = False
                if hit_stop or hit_target:
                    if sim_params.get("target_stop_mode", "immediate_exit") == "immediate_exit":
                        trade_events.append({
                            "row": i,
                            "date": row["Date"],
                            "trend_id": active_trend["trend_id"],
                            "event": "Stop" if hit_stop else "Target",
                            "exit_price": stop_p if hit_stop else target_p,
                            "side": active_trend["side"],
                            "vol_zone": active_trend["vol_zone"],
                            "Exit_Method": "Stop" if hit_stop else "Target",
                            "Target_Price": target_p,
                            "Stop_Price": stop_p,
                            "TS_Enabled": True
                        })
                        df_sim.at[i, "Trend_Active"] = False
                        df_sim.at[i, "Exit_Price"] = stop_p if hit_stop else target_p
                        df_sim.at[i, "Exit_Date"] = row["Date"]
                        active_trend = None
                        continue
                    else:
                        trade_events.append({
                            "row": i,
                            "date": row["Date"],
                            "trend_id": active_trend["trend_id"],
                            "event": "Stop" if hit_stop else "Target",
                            "exit_price": stop_p if hit_stop else target_p,
                            "side": active_trend["side"],
                            "vol_zone": active_trend["vol_zone"],
                            "Exit_Method": "Stop" if hit_stop else "Target",
                            "Target_Price": target_p,
                            "Stop_Price": stop_p,
                            "TS_Enabled": True
                        })
                        active_trend["trade_closed"] = True
            # If trade is still active, check the continuation condition.
            if active_trend is not None and not (active_trend.get("TS_Enabled") and 
               sim_params.get("target_stop_mode", "immediate_exit") == "mark_exit_only" and active_trend.get("trade_closed", False)):
                if active_trend["side"] == "Long":
                    cont_valid = evaluate_unified_condition(row, cont_long, df)
                else:
                    cont_valid = evaluate_unified_condition(row, cont_short, df)
            else:
                cont_valid = True
            if not cont_valid:
                trade_events.append({
                    "row": i,
                    "date": row["Date"],
                    "trend_id": active_trend["trend_id"],
                    "event": "Trend Closure",
                    "exit_price": row["Open"],
                    "side": active_trend["side"],
                    "vol_zone": active_trend["vol_zone"],
                    "Exit_Method": "Trend Closure",
                    "Target_Price": active_trend.get("target_price"),
                    "Stop_Price": active_trend.get("stop_price"),
                    "TS_Enabled": active_trend.get("TS_Enabled", False)
                })
                df_sim.at[i, "Trend_Active"] = False
                df_sim.at[i, "Exit_Price"] = row["Open"]
                df_sim.at[i, "Exit_Date"] = row["Date"]
                active_trend = None
            else:
                df_sim.at[i, "Trend_Active"] = True
                df_sim.at[i, "Trend_ID"] = active_trend["trend_id"]
                df_sim.at[i, "Entry_Price"] = active_trend["entry_price"]
                df_sim.at[i, "Entry_Date"] = active_trend["entry_date"]

        # End while loop for row i.
    
    # Close any active trade at the end.
    if active_trend is not None:
        trade_events.append({
            "row": nrows-1,
            "date": df_sim.at[nrows-1, "Date"],
            "trend_id": active_trend["trend_id"],
            "event": "Trend Closure",
            "exit_price": df_sim.at[nrows-1, "Open"],
            "side": active_trend["side"],
            "vol_zone": active_trend["vol_zone"],
            "Exit_Method": "Trend Closure",
            "Target_Price": active_trend.get("target_price"),
            "Stop_Price": active_trend.get("stop_price"),
            "TS_Enabled": active_trend.get("TS_Enabled", False)
        })
        df_sim.at[nrows-1, "Trend_Active"] = False
        df_sim.at[nrows-1, "Exit_Price"] = df_sim.at[nrows-1, "Open"]
        df_sim.at[nrows-1, "Exit_Date"] = df_sim.at[nrows-1, "Date"]
        active_trend = None

    # Normalize trade events into a trade list.
    normalized_trades = []
    for ev in trade_events:
        if ev["event"] in ["Trend Closure", "Stop", "Target"]:
            entry_ev = next((x for x in trade_events if x["trend_id"] == ev["trend_id"] and x["event"] == "New Trend Entry"), None)
            if entry_ev:
                eprice = entry_ev["entry_price"]
                xprice = ev["exit_price"]
                direction = 1 if ev["side"] == "Long" else -1
                fixed_size = entry_ev["fixed_size"]
                pnl_price = (xprice - eprice) * direction * fixed_size
                ticks = pnl_price / sim_params["tick_size"] if sim_params["tick_size"] != 0 else np.nan
                pnl_curr = ticks * sim_params["tick_value"]
                normalized_trades.append({
                    "Entry_Date": entry_ev["date"],
                    "Entry_Price": eprice,
                    "Direction": direction,
                    "Volatility_Zone": ev["vol_zone"],
                    "Trend_ID": ev["trend_id"],
                    "Entry_No": 1,
                    "Exit_Price": xprice,
                    "Exit_Method": ev.get("Exit_Method", "Trend Closure"),
                    "Exit_Date": ev["date"],
                    "PnL_Price": pnl_price,
                    "PnL_Currency": pnl_curr,
                    "Entry_Size": fixed_size,
                    "Trade_Label": entry_ev.get("Trade_Label", "Initial")
                })
    
    if sim_params.get("output_type", "both") == "trade_list":
        return None, pd.DataFrame(normalized_trades)
    elif sim_params.get("output_type", "both") == "final_df":
        return df_sim, None
    else:
        return df_sim, pd.DataFrame(normalized_trades)