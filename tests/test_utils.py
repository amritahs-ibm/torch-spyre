from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from codegen.utils.template_tools import (
    parse_returns,
    parse_single_argument,
    parse_func_schema,
)


class TestParsereturns(TestCase):
    def test_single_tensor(self):
        assert parse_returns("Tensor") == [{"type": "Tensor"}]

    def test_single_tensor_with_alias(self):
        assert parse_returns("Tensor(a!)") == [{"type": "Tensor(a!)"}]

    def test_tuple_unnamed(self):
        assert parse_returns("(Tensor, Tensor, Tensor)") == [
            {"type": "Tensor"},
            {"type": "Tensor"},
            {"type": "Tensor"},
        ]

    def test_tuple_with_aliases_unnamed(self):
        # e.g. native_batch_norm.out -> (Tensor(a!), Tensor(b!), Tensor(c!))
        assert parse_returns("(Tensor(a!), Tensor(b!), Tensor(c!))") == [
            {"type": "Tensor(a!)"},
            {"type": "Tensor(b!)"},
            {"type": "Tensor(c!)"},
        ]

    def test_tuple_with_aliases_named(self):
        # e.g. min.dim -> (Tensor(a!) min, Tensor(b!) max)
        assert parse_returns("(Tensor(a!) min, Tensor(b!) max)") == [
            {"type": "Tensor(a!)", "name": "min"},
            {"type": "Tensor(b!)", "name": "max"},
        ]

    def test_empty_return(self):
        # e.g. _validate_compressed_sparse_indices -> ()
        assert parse_returns("()") == []

    def test_whitespace_handling(self):
        assert parse_returns("  Tensor  ") == [{"type": "Tensor"}]


class TestParseSingleArgument(TestCase):
    # Basic types
    def test_plain_tensor(self):
        # e.g. "Tensor self"
        assert parse_single_argument("Tensor self") == {
            "type": "Tensor",
            "name": "self",
        }

    def test_plain_int(self):
        # e.g. "int blank"
        assert parse_single_argument("int blank") == {"type": "int", "name": "blank"}

    def test_plain_float(self):
        # e.g. "float momentum"
        assert parse_single_argument("float momentum") == {
            "type": "float",
            "name": "momentum",
        }

    def test_plain_bool(self):
        # e.g. "bool train"
        assert parse_single_argument("bool train") == {"type": "bool", "name": "train"}

    def test_scalar(self):
        # e.g. "Scalar alpha"
        assert parse_single_argument("Scalar alpha") == {
            "type": "Scalar",
            "name": "alpha",
        }

    def test_symint(self):
        # e.g. "SymInt hidden_size"
        assert parse_single_argument("SymInt hidden_size") == {
            "type": "SymInt",
            "name": "hidden_size",
        }

    # Optional types (trailing ?)
    def test_optional_tensor(self):
        # e.g. "Tensor? gradient"
        assert parse_single_argument("Tensor? gradient") == {
            "type": "Tensor?",
            "name": "gradient",
        }

    def test_optional_int(self):
        # e.g. "int? min"
        assert parse_single_argument("int? min") == {"type": "int?", "name": "min"}

    def test_optional_scalar_type(self):
        # e.g. "ScalarType? dtype"
        assert parse_single_argument("ScalarType? dtype") == {
            "type": "ScalarType?",
            "name": "dtype",
        }

    def test_optional_memory_format(self):
        # e.g. "MemoryFormat? memory_format"
        assert parse_single_argument("MemoryFormat? memory_format") == {
            "type": "MemoryFormat?",
            "name": "memory_format",
        }

    # Array types
    def test_tensor_list(self):
        # e.g. "Tensor[] inputs"
        assert parse_single_argument("Tensor[] inputs") == {
            "type": "Tensor[]",
            "name": "inputs",
        }

    def test_int_array(self):
        # e.g. "int[] input_lengths"
        assert parse_single_argument("int[] input_lengths") == {
            "type": "int[]",
            "name": "input_lengths",
        }

    def test_symint_array(self):
        # e.g. "SymInt[] size"
        assert parse_single_argument("SymInt[] size") == {
            "type": "SymInt[]",
            "name": "size",
        }

    def test_sized_array(self):
        # e.g. "bool[4] output_mask"
        assert parse_single_argument("bool[4] output_mask") == {
            "type": "bool[4]",
            "name": "output_mask",
        }

    def test_optional_symint_array(self):
        # e.g. "SymInt[]? size"
        assert parse_single_argument("SymInt[]? size") == {
            "type": "SymInt[]?",
            "name": "size",
        }

    # Alias annotations
    def test_mutable_alias(self):
        # e.g. "Tensor(a!) self"
        assert parse_single_argument("Tensor(a!) self") == {
            "type": "Tensor(a!)",
            "name": "self",
        }

    def test_read_alias(self):
        # e.g. "Tensor(a) self"
        assert parse_single_argument("Tensor(a) self") == {
            "type": "Tensor(a)",
            "name": "self",
        }

    # Default values
    def test_default_none(self):
        # e.g. "Tensor? gradient=None"
        assert parse_single_argument("Tensor? gradient=None") == {
            "type": "Tensor?",
            "name": "gradient",
            "default": "None",
        }

    def test_default_bool_true(self):
        # e.g. "bool count_include_pad=True"
        assert parse_single_argument("bool count_include_pad=True") == {
            "type": "bool",
            "name": "count_include_pad",
            "default": "True",
        }

    def test_default_int(self):
        # e.g. "int self_num_batch_dims=0"
        assert parse_single_argument("int self_num_batch_dims=0") == {
            "type": "int",
            "name": "self_num_batch_dims",
            "default": "0",
        }

    def test_default_float(self):
        # e.g. "float rtol=1e-05"
        assert parse_single_argument("float rtol=1e-05") == {
            "type": "float",
            "name": "rtol",
            "default": "1e-05",
        }

    def test_default_scalar(self):
        # e.g. "Scalar alpha=1"
        assert parse_single_argument("Scalar alpha=1") == {
            "type": "Scalar",
            "name": "alpha",
            "default": "1",
        }

    def test_default_empty_list(self):
        # e.g. "int[1] stride=[]"
        assert parse_single_argument("int[1] stride=[]") == {
            "type": "int[1]",
            "name": "stride",
            "default": "[]",
        }

    # Invalid / edge cases
    def test_returns_none_for_invalid(self):
        # No name — should return None
        assert parse_single_argument("Tensor") == {}

    def test_returns_none_for_empty(self):
        assert parse_single_argument("") == {}


class TestParseFuncSchema(TestCase):
    # Operator name variants
    def test_no_overload(self):
        # e.g. _use_cudnn_ctc_loss(...) -> bool
        result = parse_func_schema(
            "_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool"
        )
        assert result["operator_name"] == "_use_cudnn_ctc_loss"
        assert result["overload_name"] == ""

    def test_out_overload(self):
        # e.g. addmv.out(...) -> Tensor(a!)
        result = parse_func_schema(
            "addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"
        )
        assert result["operator_name"] == "addmv"
        assert result["overload_name"] == "out"

    def test_non_out_overload(self):
        # e.g. bernoulli.p(...) -> Tensor
        result = parse_func_schema(
            "bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor"
        )
        assert result["operator_name"] == "bernoulli"
        assert result["overload_name"] == "p"

    def test_inplace_operator(self):
        # e.g. addmv_(...) -> Tensor(a!)  — trailing _ in operator name
        result = parse_func_schema(
            "addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)"
        )
        assert result["operator_name"] == "addmv_"
        assert result["overload_name"] == ""

    # Return type variants
    def test_single_tensor_return(self):
        result = parse_func_schema(
            "allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool"
        )
        assert result["returns"] == [{"type": "bool"}]

    def test_single_tensor_with_alias_return(self):
        # e.g. -> Tensor(a!)
        result = parse_func_schema(
            "addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)"
        )
        assert result["returns"] == [{"type": "Tensor(a!)"}]

    def test_empty_return(self):
        # e.g. _backward(...) -> ()
        result = parse_func_schema(
            "_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()"
        )
        assert result["returns"] == []

    def test_five_tuple_return(self):
        # e.g. _cudnn_rnn(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
        result = parse_func_schema(
            "_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        )
        assert len(result["returns"]) == 5
        assert all(r == {"type": "Tensor"} for r in result["returns"])

    def test_named_return_with_alias(self):
        # e.g. lstsq.X(...) -> (Tensor(a!) solution, Tensor(b!) QR)
        result = parse_func_schema(
            "lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)"
        )
        assert result["returns"] == [
            {"type": "Tensor(a!)", "name": "solution"},
            {"type": "Tensor(b!)", "name": "QR"},
        ]

    # Argument variants
    def test_keyword_only_args(self):
        # *, separates positional from keyword-only args
        result = parse_func_schema(
            "addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"
        )
        names = [a["name"] for a in result["arguments"]]
        assert names == ["self", "mat", "vec", "beta", "alpha", "out"]

    def test_no_arguments(self):
        # e.g. a function with only keyword-only args and no positional ones
        result = parse_func_schema(
            "_make_dep_token(*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"
        )
        names = [a["name"] for a in result["arguments"]]
        assert names == ["dtype", "layout", "device", "pin_memory", "memory_format"]

    def test_sized_array_arg(self):
        # e.g. bool[3] output_mask, bool[4] output_mask
        result = parse_func_schema(
            "_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)"
        )
        output_mask = next(a for a in result["arguments"] if a["name"] == "output_mask")
        assert output_mask["type"] == "bool[3]"

    def test_default_empty_list_arg(self):
        # e.g. int[1] stride=[]
        result = parse_func_schema(
            "avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor"
        )
        stride = next(a for a in result["arguments"] if a["name"] == "stride")
        assert stride["default"] == "[]"

    def test_schema_string_preserved(self):
        func_str = "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        result = parse_func_schema(func_str)
        assert result["schema_string"] == func_str

    def test_schema_order_arguments_is_copy(self):
        # schema_order_arguments should be equal in content but a separate list
        result = parse_func_schema(
            "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        )
        assert result["arguments"] == result["schema_order_arguments"]
        assert result["arguments"] is not result["schema_order_arguments"]


if __name__ == "__main__":
    run_tests()
