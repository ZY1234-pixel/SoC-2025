import os
from typing import List
from input.load import load_from_folder, load_from_file
from settings import settings
import click

class CLILoader:
    def __init__(self, filepath: str, cli_options: dict, highres: bool = False):
        self.page_range = cli_options.get("page_range")
        if self.page_range:
            self.page_range = self.parse_range_str(self.page_range)
        self.filepath = filepath
        # self.config = cli_options
        self.save_images = True
        self.debug = True
        self.output_dir = os.path.join(settings.RESULT_DIR)
        self.load(highres)


    def load(self, highres: bool = False):
        highres_images = None
        if os.path.isdir(self.filepath):
            images, names = load_from_folder(self.filepath, self.page_range)
            folder_name = os.path.basename(self.filepath)
            if highres:
                highres_images, _ = load_from_folder(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)
        else:
            images, names = load_from_file(self.filepath, self.page_range)
            folder_name = os.path.basename(self.filepath).split(".")[0]
            if highres:
                highres_images, _ = load_from_file(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)


        self.images = images
        self.highres_images = highres_images
        self.names = names

        self.result_path = os.path.abspath(os.path.join(self.output_dir, folder_name))
        os.makedirs(self.result_path, exist_ok=True)


    @staticmethod
    def parse_range_str(range_str: str) -> List[int]:
        range_lst = range_str.split(",")
        page_lst = []
        for i in range_lst:
            if "-" in i:
                start, end = i.split("-")
                page_lst += list(range(int(start), int(end) + 1))
            else:
                page_lst.append(int(i))
        page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
        return page_lst

    def load(self, highres: bool = False):
        highres_images = None
        if os.path.isdir(self.filepath):
            images, names = load_from_folder(self.filepath, self.page_range)
            folder_name = os.path.basename(self.filepath)
            if highres:
                highres_images, _ = load_from_folder(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)
        else:
            images, names = load_from_file(self.filepath, self.page_range)
            folder_name = os.path.basename(self.filepath).split(".")[0]
            if highres:
                highres_images, _ = load_from_file(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)

        self.images = images
        self.highres_images = highres_images
        self.names = names

        self.result_path = os.path.abspath(os.path.join(self.output_dir, folder_name))
        os.makedirs(self.result_path, exist_ok=True)