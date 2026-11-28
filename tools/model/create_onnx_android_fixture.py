#!/usr/bin/env python3
"""Generate a tiny synthetic ONNX Identity model for Android runtime tests.

This fixture is a software test artifact only. It contains no user data and is
not a Tata language model or training corpus.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "tata_synthetic_identity", [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="tata-ai-test-fixture",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    args.output.write_bytes(model.SerializeToString())


if __name__ == "__main__":
    main()
