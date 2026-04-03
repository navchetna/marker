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
from marker.output import output_exists, save_output

from tree_parser.tree import Tree
from tree_parser.treeparser import TreeParser

configure_logging()
logger = get_logger()


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown.")
@click.argument("fpath", type=str)
@click.option(
    "--user",
    required=True,
    type=str,
    help="The user name or ID to determine the output directory."
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Base output directory. Defaults to ~/user."
)
@ConfigParser.common_options
def convert_single_cli(fpath: str, user: str, output_dir: str, **kwargs):
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
    tree = Tree(fpath, user_param=user, output_dir=output_dir)
    tree_parser = TreeParser(user, output_dir=output_dir)

    # Generate markdown with marker
    filename = tree_parser.get_filename(fpath)
    output_path = os.path.join(tree_parser.OUTPUT_DIR, filename)
    if not output_exists(output_path, filename):
        rendered = converter(fpath)
        os.makedirs(output_path, exist_ok=True)
        save_output(rendered, output_path, filename)
        logger.info(f"Saved markdown to {output_path}")

    # Build tree structure (TOC via docling + parse markdown)
    tree_parser.populate_tree(tree)

    tree_parser.generate_output_text(tree)
    tree_parser.generate_output_json(tree)
    logger.info(f"Total time: {time.time() - start}")
