import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from qdrant_client import QdrantClient
from sqlalchemy import JSON

# load API keys from globally-availabe .env file
load_dotenv(dotenv_path='../.env', override=True)

class VectorDatabase:
    """Contains all methods for building and using vector databases.
    """

    def index(self,test: int = 1)->JSON:
        """_summary_

        Args:
            test (int, optional): _description_. Defaults to 1.

        Returns:
            JSON: _description_
        """
        return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

    def getContexts(self,):
        """Here's a summary of the work.

        ## GET arguments
        course name (optional) str
            A json response with TBD fields.
            
        Returns
        -------
        JSON
            A json response with TBD fields.

        Raises
        ------
        Exception
            Testing how exceptions are handled.
        """
        # todo: best way to handle optional arguments?
        try:
            language: str = request.args.get('course_name')
        except Exception as e:
            print("No course name provided.")
        try:
            language: str = request.args.get('course_name')
        except Exception as e:
            print("No course name provided.")

        language: str = request.args.get('course_name')
        response:str = jsonify({"language": f"You said: {language}"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        if language == 'error':
            raise Exception('This is an error message!')
        return response

    def sayhi(self, cool: str = 'cool'):
        """Here's what it does
        
        Parameters
        ----------
        cool : str, optional 

        Returns
        -------
        JSON
            A simple json response.
        
        Example usage
        -------------
        First mode, *buffer* is `None`:
        ```python
        sayhi(cool='cool')
        {"language": "Hi there: cool"}
        ```
        """
        language = request.args.get('language')
        response = jsonify({"language": f"Hi there: {language}"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


    def get_contexts(self,):
        contexts = {'language': 'python', 'framework': 'Flask'}
        response = jsonify(contexts)
        return response



    def getqdrant(self,):
        qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        return 'Placeholder return'


# if __name__ == '__main__':
    # app.run(debug=True, port=os.getenv("PORT", default=5000))
