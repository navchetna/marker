import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import time
import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output

import sys
from pathlib import Path
pdf_parsing_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pdf_parsing_root))

from parsers.tree import Tree
from parsers.treeparser import TreeParser

configure_logging()
logger = get_logger()


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown.")
@click.argument("fpath", type=str)
@ConfigParser.common_options
def convert_single_cli(fpath: str, **kwargs):
    models = create_model_dict()
    start = time.time()
    config_parser = ConfigParser(kwargs)

    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    tree = Tree(fpath)
    tree_parser = TreeParser()
    tree_parser.populate_tree(tree, converter)

    tree_parser.generate_output_text(tree)
    tree_parser.generate_output_json(tree)
    logger.info(f"Total time: {time.time() - start}")

def convert_pdf_to_markdown(fpath: str, config_kwargs: dict | None = None) -> str:
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict

    cfg = dict(config_kwargs or {})
    cfg.setdefault("output_format", "markdown")
    models = create_model_dict()
    config_parser = ConfigParser(cfg)

    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(fpath)
    return rendered
