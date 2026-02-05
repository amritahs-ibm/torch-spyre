# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jinja2 import Environment, FileSystemLoader

from codegen.utils.shape_extractor import infer_output_shape_stride

import regex as re
from typing import List, Dict, Any


def extract_scalar_arg_names(schema_string: str) -> List[str]:
    """
    Extract scalar argument names from PyTorch operator schema.

    Examples:
        extract_scalar_arg_names("aten::add.Tensor(Tensor self, *, Scalar alpha=1) -> Tensor")
        ['alpha']
    """
    # Extract arguments from parentheses
    match = re.search(r"\((.*?)\)\s*->", schema_string)
    if not match:
        return []
    args_str = match.group(1)
    # Find all Scalar arguments with their names
    # Pattern: Scalar (optionally ?) followed by whitespace and name
    pattern = r"Scalar\??[\s]+([a-zA-Z_][a-zA-Z0-9_]*)"
    all_scalar_names = re.findall(pattern, args_str)
    # Filter out alpha and beta
    return [name for name in all_scalar_names if name not in ["alpha", "beta"]]


def get_args_with_default_vals(schema_string):
    """
    Extract keyword-only argument names (those after '*')
    from a PyTorch operator schema string.

    In PyTorch schemas, arguments after '*' are keyword-only and typically have
    default values. This function identifies and returns the names of those arguments.

    Args:
        schema_string (str): PyTorch operator schema string in the format:
                "op_name(arg1, arg2, *, kwarg1=default1, kwarg2=default2) -> return_type"

    Returns:
        list[str]: List of keyword-only argument names (without defaults or types)

    Examples:
        schema = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        ['alpha']

        schema = "aten::clamp(Tensor self, *, Scalar? min=None, Scalar? max=None) -> Tensor"
        ['min', 'max']

        schema = "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        []

        schema = "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *,
                            Scalar beta=1, Scalar alpha=1) -> Tensor"
        ['beta', 'alpha']

    """
    # Extract everything inside parentheses
    inside = re.search(r"\((.*)\)", schema_string).group(1)
    # Split by comma and clean spacing
    parts = [p.strip() for p in inside.split(",")]
    # Find the index of "*"
    if "*" in parts:
        parts = parts[parts.index("*") + 1 :]
    else:
        parts = []
    args_with_def_vals = []
    for p in parts:
        name = p.split()[-1]  # last token: alpha=1
        name = name.split("=")[0]  # remove default: alpha
        args_with_def_vals.append(name)
    return args_with_def_vals


def format_python_signature(arguments):
    """
    Convert argument list to Python function signature string.

    Example:
        [{'name': 'self', 'type': 'Tensor'}, {'name': 'mat2', 'type': 'Tensor'}]
        -> "self: torch.Tensor, mat2: torch.Tensor"
    """
    sig_parts = []
    for arg in arguments:
        type_str = convert_cpp_type_to_python(arg["type"])
        # Handle default values if present
        if "default" in arg and arg["default"] is not None and arg["default"] != "":
            default_val = format_default_value(arg["default"])
            sig_parts.append(f"{arg['name']}: {type_str} = {default_val}")
        elif "out" in arg["name"]:
            sig_parts.append(f"{arg['name']}: {type_str} = None")
        else:
            sig_parts.append(f"{arg['name']}: {type_str}")
    return ", ".join(sig_parts)


def format_default_value(default_val):
    """
    Format default value for Python.

    Examples:
        c10::nullopt -> None
        true -> True
        false -> False
        1.0 -> 1.0
    """
    if default_val in ["c10::nullopt", "nullptr", "::std::nullopt"]:
        return "None"
    elif default_val == "true":
        return "True"
    elif default_val == "false":
        return "False"
    else:
        return str(default_val)


def format_python_return_type(returns):
    """
    Convert return type to Python type annotation.

    Example:
        [{'type': 'Tensor'}] -> "torch.Tensor"
        [{'type': 'Tensor'}, {'type': 'Tensor'}] -> "tuple[torch.Tensor, torch.Tensor]"
    """
    if not returns:
        return "None"

    if len(returns) == 1:
        return convert_cpp_type_to_python(returns[0]["type"])

    # Multiple returns - use tuple
    types = [convert_cpp_type_to_python(r["type"]) for r in returns]
    return f"tuple[{', '.join(types)}]"


def convert_cpp_type_to_python(cpp_type):
    """Convert C++ type to Python type annotation."""
    # Remove const, &, * modifiers
    clean_type = re.sub(r"\([a-z]?!?\)", "", cpp_type)
    clean_type = (
        clean_type.replace("at::", "")
        .replace("const", "")
        .replace("&", "")
        .replace("*", "")
        .strip()
    )
    type_mapping = {
        "bool": "bool",
        "double": "float",
        "Dimname[]": "list[str]",
        "Dimname": "str",
        "int[1]?": "int | None",
        "int[]": "list[int]",
        "SymInt[]": "list[int]",
        "SymInt": "int",
        "str?": "str | None",
        "str[1]": "str",
        "ScalarType?": "torch.dtype | None",
        "ScalarType": "torch.dtype",
        "Scalar": "Union[int, float, bool, complex]",
        "Tensor[]": "list[Tensor]",
        "Tensor": "torch.Tensor",
    }

    for cpp, py in type_mapping.items():
        if cpp in clean_type:
            clean_type = clean_type.replace(cpp, py)

    # Handle optionals
    if "optional" in cpp_type.lower() or "Optional" in cpp_type:
        clean_type = "None"

    return clean_type


def get_argument_names(arguments, schema_string):
    """
    Get comma-separated list of argument names.

    Args:
        arguments: List of argument dicts
        exclude_out: If True, exclude the 'out' parameter

    Returns:
        str: "self, mat2" or "self, mat2, alpha=alpha"
    """
    names = []
    args_with_def_vals = get_args_with_default_vals(schema_string)
    for arg in arguments:
        if arg["name"] == "out":
            continue
        if arg["name"] in args_with_def_vals:
            names.append(f"{arg['name']}={arg['name']}")
        else:
            names.append(arg["name"])
    return ", ".join(names)


def append_scalar_suffix(arg_names: str, scalar_arg_names: List[str]) -> str:
    """
    Append '_scaTensor' suffix to scalar argument names in the arg_names string.

    Args:
        arg_names: Comma-separated string of argument names
        scalar_arg_names: List of scalar argument names

    Returns:
        Modified arg_names string with scalar args suffixed

    Examples:
        append_scalar_suffix("self, other, alpha", ["other"])
        'self, other_scaTensor, alpha'
    """
    # Split arg_names into individual arguments
    args = [arg.strip() for arg in arg_names.split(",")]
    # Modify arguments that are scalars
    modified_args = []
    for arg in args:
        if arg in scalar_arg_names:
            modified_args.append(f"{arg}_scaTensor")
        else:
            modified_args.append(arg)
    # Rejoin with comma-space
    return ", ".join(modified_args)


def enhance_replacement_data(rep_data):
    """
    Add Python-specific fields to replacement data.
    This should be called in generate_replacements() for each operation.
    """
    arguments = rep_data.get("arguments", [])
    returns = rep_data.get("returns", [])
    schema_string = rep_data.get("schema_string", [])

    # Generate Python signature
    rep_data["signature_in"] = format_python_signature(arguments)
    rep_data["signature_out"] = format_python_return_type(returns)

    # Generate argument name lists
    rep_data["scalar_arg_names"] = extract_scalar_arg_names(schema_string)
    rep_data["arg_names"] = get_argument_names(arguments, schema_string)
    rep_data["arg_names"] = append_scalar_suffix(
        rep_data["arg_names"], rep_data["scalar_arg_names"]
    )

    return rep_data


def parse_arguments(args_str: str) -> List[Dict[str, Any]]:
    """
    Parse argument list from function schema.

    Example:
        Input:  "Tensor(a!) self, Tensor(b!) other, *, bool copy=False"
        Output: [
            {"name": "self", "type": "Tensor(a!)"},
            {"name": "other", "type": "Tensor(b!)"},
            {"name": "copy", "type": "bool", "default": "False"},
        ]

    """
    arguments = []
    arg_tokens = [t.strip() for t in args_str.split(",") if t.strip()]
    # Parse each argument
    for token in arg_tokens:
        if token == "*":
            # Keyword-only separator, skip
            continue

        arg_dict = parse_single_argument(token)
        if arg_dict:
            arguments.append(arg_dict)
    return arguments


def parse_single_argument(arg_str: str) -> Dict[str, Any]:
    """
    Parse a single argument token.

    Examples:
        "Tensor self" -> {type: "Tensor", name: "self"}
        "Scalar alpha=1" -> {type: "Scalar", name: "alpha", default: "1"}
        "int? dim=None" -> {type: "int?", name: "dim", default: "None"}
        "Tensor(a!) out" -> {type: "Tensor(a!)", name: "out"}
    """
    # Handle mutable annotations like Tensor(a!)
    type_match = re.match(
        r"^(\S+?(?:\([^)]*\))?)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:=(.+))?$", arg_str
    )

    if not type_match:
        return {}

    arg_type = type_match.group(1)
    arg_name = type_match.group(2)
    default_val = type_match.group(3)

    arg_dict = {
        "type": arg_type,
        "name": arg_name,
    }

    if default_val:
        arg_dict["default"] = default_val

    return arg_dict


def parse_returns(returns_str: str) -> List[Dict[str, Any]]:
    """
    Parse return type(s).

    Examples:
        "Tensor" -> [{'type': 'Tensor'}]
        "(Tensor, Tensor)" -> [{'type': 'Tensor'}, {'type': 'Tensor'}]
        "(Tensor(a!) min, Tensor(b!) max)" -> [{'type': 'Tensor(a!)', 'name': 'min'}, ...]
    """
    returns_str = returns_str.strip()

    if returns_str.startswith("(") and returns_str.endswith(")"):
        inner = returns_str[1:-1]
        if not inner.strip():  # handle -> ()
            return []
        return_tokens = [t.strip() for t in inner.split(",")]

        returns = []
        for token in return_tokens:
            match = re.match(
                r"^(\S+?(?:\([^)]*\))?)\s+([a-zA-Z_][a-zA-Z0-9_]*)$", token
            )
            if match:
                returns.append({"type": match.group(1), "name": match.group(2)})
            else:
                returns.append({"type": token})

        return returns
    else:
        return [{"type": returns_str}]


def parse_func_schema(func_string: str) -> Dict[str, Any]:
    """
    Parse a function schema from native_functions.yaml.

    Format: func_name[.overload](args) -> return_type

    Example:
        "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"

    Returns:
        dict with operator_name, overload_name, arguments, returns, schema
    """
    # Extract function name and overload
    func_match = re.match(
        r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z_][a-zA-Z0-9_]*))?", func_string
    )
    if not func_match:
        raise ValueError(f"Cannot parse function name from: {func_string}")

    operator_name = func_match.group(1)
    overload_name = func_match.group(2) or ""

    # Extract arguments and return type
    schema_match = re.search(r"\((.*?)\)\s*->\s*(.+)", func_string)
    if not schema_match:
        raise ValueError(f"Cannot parse schema from: {func_string}")

    args_str = schema_match.group(1)
    returns_str = schema_match.group(2).strip()

    # Parse arguments
    arguments = parse_arguments(args_str)

    # Parse returns
    returns = parse_returns(returns_str)

    return {
        "operator_name": operator_name,
        "overload_name": overload_name,
        "arguments": arguments,
        "schema_order_arguments": arguments.copy(),
        "returns": returns,
        "schema_string": func_string,
    }


def generate_signature_dict(replacement_dict):
    signatures = {}

    if len(replacement_dict["returns"]) == 0:
        signatures["signature_out"] = "void"
    elif len(replacement_dict["returns"]) == 1:
        signatures["signature_out"] = replacement_dict["returns"][0]["type"]
    else:
        signatures["signature_out"] = (
            f"::std::tuple<{','.join([o['type'] for o in replacement_dict['returns']])}>"
        )

    signatures["signature_in"] = ", ".join(
        [
            f"{i['type']} {i['name']}"
            for i in replacement_dict["arguments"]
            if i.get("in_signature", True)
        ]
    )

    return signatures


def generate_from_template(
    template_dir: str, template_name: str, replacement_data: dict
):
    """
    Generates a snippet from a template file by replacing keywords.

    Args:
        replacement_data (dict): A dict containing replacement data.
    """

    template_path = f"{template_name}.jinja2"  # Path of the body template file

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path)

    output = template.render(**replacement_data) + "\n\n"

    return output


def generate_replacements(
    native_functions_file, metadata, action="skip", only_req=False
):
    """
    Generates replacement data for PyTorch ops (specified in declaration and schema files)

    Args:
        all_declarations (list): list of dicts parsed from pytorch Declarations.yaml
        all_schemas (list): list of dicts parsed from pytorch RegistrationDeclarations.yaml (indices match with declarations)
        metadata (dict): dict of metadata for each operator (contains template_name and arg_mapping) parsed from Metadata.yaml
        action (str): what to do if the operator is not supported, options: 'skip', 'fallback', 'native_call'
        only_req (bool): set true to enable filtering with (dispatch=True, default=False)
    """
    import yaml

    try:
        with open(native_functions_file, "r") as f:
            native_functions = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
    replacements = []
    num_total_funcs = len(native_functions)
    num_supported_funcs = 0
    gen_op_name_list = []

    for func_entry in native_functions:
        if "func" not in func_entry:
            continue
        func_schema = func_entry["func"]
        try:
            declaration = parse_func_schema(func_schema)
        except ValueError as e:
            print(f"Warning: Failed to parse schema: {func_schema} - {e}")
            continue
        if "autogen" in func_entry:
            declaration["operator_name"] = func_entry["autogen"].split(".")[0]
        if declaration["operator_name"] in metadata:
            declaration["template_name"] = metadata[declaration["operator_name"]][
                "template_name"
            ]
            cur_metadata = metadata[declaration["operator_name"]]
            gen_op_name = declaration["operator_name"] + declaration["overload_name"]
            num_supported_funcs += 1
        else:
            cur_metadata = {
                "operator_name": declaration["operator_name"].capitalize(),
                "out_shape_stride_expr": "infer",
            }

            if action == "skip":  # skip
                # print(f"Warning: {dec['operator_name']}.{dec['overload_name']} - No metadata found, skipping...")
                continue
            else:
                if action == "fallback":  # use cpu fallback template
                    declaration["template_name"] = "fallback"
                elif action == "native":  # call aten::native
                    declaration["template_name"] = "native_call"
                else:
                    raise NotImplementedError(
                        f"{action} is not implemented, options: 'skip', 'fallback', 'native', 'auto'"
                    )

        # TODO: If first argument is not a Tensor (e.g. arange), skip.
        if len(declaration["arguments"]) > 0 and any(
            [
                t in declaration["arguments"][0]["type"]
                for t in ["int", "double", "float", "Scalar"]
            ]
        ):
            continue

        if (
            declaration["template_name"] in ["view", "view_copy"]
            and declaration["overload_name"] == "dtype"
        ):
            print(
                f"Warning: {declaration['operator_name']}.{declaration['overload_name']} - View op with dtype overload, skipping..."
            )
            continue

        if gen_op_name not in gen_op_name_list:
            declaration = generate_replacement_base_variant(declaration, cur_metadata)
            declaration = enhance_replacement_data(declaration)
            replacements.append(declaration)
            gen_op_name_list.append(gen_op_name)

        if "autogen" in func_entry:
            autogen_variants = func_entry["autogen"]
            if not isinstance(autogen_variants, list):
                autogen_variants = [autogen_variants]
                for variant in autogen_variants:
                    # Generate the out variant
                    if variant.endswith(".out") or "_out" in variant:
                        out_declaration = generate_replacement_out_variant(
                            declaration, variant, cur_metadata
                        )
                        if out_declaration:
                            out_declaration = enhance_replacement_data(out_declaration)
                            replacements.append(out_declaration)
    print(f"{num_supported_funcs} of {num_total_funcs} declarations are supported.")

    return replacements


def generate_replacement_base_variant(declaration, cur_metadata):
    """
    Process a declaration of each operator, and add template_data and normalize
    boolean defaults.

    Args:
    declaration (dict): A dictionary describing a PyTorch operator, including
                            its name, overload name, and arguments.
    cur_metadata (dict): curret_metadata obtained from Metadata.yaml file

    Examples:
    declaration: {'operator_name': 'abs',
                  'overload_name': '',
                  'arguments': [{'type': 'Tensor', 'name': 'self'}],
                  'schema_order_arguments': [{'type': 'Tensor', 'name': 'self'}],
                  'returns': [{'type': 'Tensor'}],
                  'schema_string': 'abs(Tensor self) -> Tensor',
                  'template_name': 'base'}
    cur_metadata: {'operator_name': 'abs', 'template_name': 'base'}

    Returns:
    {'operator_name': 'abs',
     'overload_name': '',
     'arguments': [{'type': 'Tensor', 'name': 'self'}],
     'schema_order_arguments': [{'type': 'Tensor', 'name': 'self'}],
     'returns': [{'type': 'Tensor'}],
     'schema_string': 'abs(Tensor self) -> Tensor',
     'template_name': 'base',
     'template_data': {'op_name': 'abs_default', 'op_label': '"Abs"', 'reg_name': '"abs"', 'torch_prefix': 'torch', 'torch_func_name': 'abs'},
     'signature_out': 'Tensor',
     'signature_in': 'Tensor self',
     'out_shape_stride_expr': 'bypass'}

    """

    declaration["template_data"] = {
        "op_name": declaration["operator_name"]
        + "_"
        + (declaration["overload_name"] if declaration["overload_name"] else "default"),
        "op_label": f'"{declaration["operator_name"].capitalize()}"',
        "reg_name": f'"{declaration["operator_name"]}.{declaration["overload_name"]}"'
        if declaration["overload_name"]
        else f'"{declaration["operator_name"]}"',
        "torch_prefix": cur_metadata.get("torch_prefix", "torch"),
        "torch_func_name": cur_metadata.get(
            "torch_func_name", declaration["operator_name"]
        ),
    }

    signatures = generate_signature_dict(declaration)
    declaration |= signatures

    for dec_arg in declaration["arguments"]:
        if "default" in dec_arg and isinstance(dec_arg["default"], bool):
            dec_arg["default"] = str(dec_arg["default"]).lower()

    # unless there is a provided out_shape_stride_expr method, we will skip output shape and stride inference (first input will be used directly)
    declaration["out_shape_stride_expr"] = cur_metadata.get(
        "out_shape_stride_expr", "bypass"
    )

    # if the template is base and out_shape_stride_expr is infer, we can try auto shape inference
    if (
        declaration["template_name"] == "base"
        and declaration["out_shape_stride_expr"] == "infer"
    ):
        output_shape_stride_list, bypass_flag = infer_output_shape_stride(declaration)
        if output_shape_stride_list is not None:
            if bypass_flag:
                declaration["out_shape_stride_expr"] = "bypass"
                # Output shape inference is not necessary
                pass
            else:
                # inferred symbolic representation that will be used in the template
                for i, output_shape_stride in enumerate(output_shape_stride_list):
                    if output_shape_stride:
                        declaration["returns"][i]["shape"] = output_shape_stride[
                            "shape"
                        ]
                        declaration["returns"][i]["stride"] = output_shape_stride[
                            "stride"
                        ]
    return declaration


def generate_replacement_out_variant(base_declaration, variant_name, cur_metadata):
    """
    Generate an .out variant from a base declaration.
    Args:
        base_declaration: The base function declaration
        variant_name: The variant name (e.g., "repeat.out")
        cur_metadata: Metadata for the operator

    Returns:
        New declaration for the .out variant
    """
    import copy

    # Extract overload name from variant (e.g., "repeat.out" -> "out")
    overload = variant_name.split(".")[-1] if "." in variant_name else variant_name

    # Create a copy of the base declaration
    out_declaration = copy.deepcopy(base_declaration)

    # Update overload name
    out_declaration["overload_name"] = overload

    # Add 'out' parameter to arguments
    # The out parameter should match the return type
    if out_declaration["returns"]:
        return_type = out_declaration["returns"][0]["type"]

        # Create out parameter
        out_param = {
            "type": return_type,
            "name": "out",
        }

        # Add out parameter at the end
        out_declaration["arguments"].append(out_param)
        out_declaration["schema_order_arguments"].append(out_param)

    # Update schema string to include .out overload
    original_schema = out_declaration["schema_string"]
    # Insert .out into the schema
    out_declaration["schema_string"] = original_schema.replace(
        f"{out_declaration['operator_name']}(",
        f"{out_declaration['operator_name']}.{overload}(",
        1,
    )

    # Also add out parameter to schema
    # Find the closing parenthesis before ->
    schema_parts = out_declaration["schema_string"].split("->")
    if len(schema_parts) == 2:
        args_part = schema_parts[0].rstrip()
        return_part = schema_parts[1].strip()

        # Add out parameter before closing parenthesis
        if args_part.endswith(")"):
            args_part = args_part[:-1] + f", {return_type} out)"

        out_declaration["schema_string"] = f"{args_part} -> {return_part}"

    # Don't re-process metadata, just update the template_data
    out_declaration["template_data"] = {
        "op_name": out_declaration["operator_name"] + "_" + overload,
        "op_label": f'"{out_declaration["operator_name"].capitalize()}"',
        "sendnn_func_name": cur_metadata.get("sendnn_func_name", ""),
        "reg_name": f'"{out_declaration["operator_name"]}.{overload}"',
        "torch_prefix": cur_metadata.get("torch_prefix", "torch"),
        "torch_func_name": cur_metadata.get(
            "torch_func_name", out_declaration["operator_name"]
        ),
    }

    return out_declaration
