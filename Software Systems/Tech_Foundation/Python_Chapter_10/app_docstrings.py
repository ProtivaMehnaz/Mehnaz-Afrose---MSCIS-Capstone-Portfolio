# Example of a Docstring. Nothing for you code, just be familiar with best practices of writing docstrings documentation.
# Best practice is to include a brief description, parameters, and return values if any.
""" This module provides functions to convert a PDF to text or png image
"""


class Converter:
    """ A simple converter for converting PDFs to text."""

    def convert_to_text(self, path):
        """
        Convert the given PDF to text.

        Parameters: 
        path (str): The path to a PDF file

        Returns: 
        str: The content of a PDF file as text
        """
        print("pdf2text")

    def convert_to_image(self, path):
        """
        Convert the given PDF to a png image

        Parameters: 
        path (str): The path to a PDF file

        Returns: 
        str: The content of a PDF file as a png image
        """
        print("pdf2image")
