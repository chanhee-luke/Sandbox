#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from sandbox.utils.misc import get_logger
from sandbox.translate.translator import build_translator

import sandbox.input
import sandbox.translate
import sandbox
import sandbox.model_builder
import sandbox.modules
import sandbox.utils.opts


def main(opt):
    translator = build_translator(opt, report_score=True, logger=logger)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sandbox.utils.opts.add_md_help_argument(parser)
    sandbox.utils.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = get_logger(opt.log_file)
    main(opt)
