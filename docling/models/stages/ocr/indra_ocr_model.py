# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import IndraOcrOptions, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder


class IndraOcrModel(BaseOcrModel):
    """Indra OCR integration using its in-memory image API."""

    multiple_languages = True

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: IndraOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: IndraOcrOptions
        self.scale = self.options.scale

        if self.enabled:
            try:
                import indra_ocr
            except ImportError as error:
                raise ImportError(
                    "indra-ocr is not installed. Please install it via "
                    "`pip install indra-ocr` to use this OCR engine."
                ) from error

            model_path = (
                str(artifacts_path / "IndraOcr") if artifacts_path is not None else None
            )
            self.reader = indra_ocr.load(
                path=model_path,
                threads=self.options.threads,
            )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells = []
                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue
                    image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    ).convert("L")
                    result = self.reader.ocr(numpy.asarray(image))
                    cells = [
                        TextCell(
                            index=ix,
                            text=line.text,
                            orig=line.text,
                            from_ocr=True,
                            confidence=line.conf,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    coord=(
                                        line.box[0] / self.scale + ocr_rect.l,
                                        line.box[1] / self.scale + ocr_rect.t,
                                        line.box[2] / self.scale + ocr_rect.l,
                                        line.box[3] / self.scale + ocr_rect.t,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                        for ix, line in enumerate(result.lines)
                        if line.text
                    ]
                    all_ocr_cells.extend(cells)

                self.post_process_cells(all_ocr_cells, page, conv_res)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return IndraOcrOptions
