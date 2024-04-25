from AWS_credentials import *

import boto3


class BedrockLLM:

    @staticmethod
    def get_bedrock_client():
        """
        This function will return bedrock client.
        """
        bedrock_client = boto3.client(
            'bedrock',
            region_name=AWS_REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            aws_session_token="IQoJb3JpZ2luX2VjEFgaCXVzLWVhc3QtMSJIMEYCIQCyOaIrHCax2JTb+5IJoFGps/mLSK+TG7hWQKPxdIlzrAIhAMIxeBikXE4yfBhQdDhFQH0dWi6A/aO3mnSomUi9mtFJKo8DCKH//////////wEQABoMNDcxMTEyOTY3MjIwIgzLLy2pkiU+o5M8Q3Qq4wLWTX34XQCsCiOc1Z69fCuAg0Pu0nZvpxSY9uVRBFeGqC+oqoJIUS3YPNDbGqrhOHB/6xLfXalVMl7gQc5nHcZrQEjhUGhn/9/ZSAcP7wKQdLZqjRtbqdnB3GteaxSUaH77OIabpJmwb9MMf3socZdxeBBmgXCXcid6sG5MXOI8l4CeC4XwHZqb1nVcEhsoVgXem96nWmwHuAHIEeBg4HN2kwl/pnQE6LhGoik43uHPsLOw82XDE9ABZ4HIYxtJNZLI6MsR9nxM69Q2Nl3iqg5WqYPoP1za45AfYfjL3UqNn/0BZnIZULcxINuPiH/u/1SBfPfSijF61SvsDUq177vm44FwaDA+jQdHL9aFYZ0YjGUjS/9TKjMqaWGBHa0MjpW2rjLqg0L65+3N7djtn9jNUQpm20r5hJaX1UUoHjwds1WVTFmCZ9ckdUeHDTM29F/svkHIQxrsJwF9zisgWnKqlBaNMKSSqLEGOqUBx19pV4TEdu5jWRduX8gaQu8Y7pb11JJwPQzjl57qyz/LevPZWrxm3xdXiSabaOKzzkXKXLZG2x87OcjmYhk58RVLmJ6zaEq2cNfJBFaiJkOHMvkEWhZ4jLh7SLal4/ZZybTyw0jjqMUR3ui8JQUt5MmY9Ms7p1CBJvIcCUY7Ea8cV2Xt9RdVqfJVj9ort0NyKghKU6lDlHOVrP5owgHfJEIiH9tT"

        )

        return bedrock_client

    @staticmethod
    def get_bedrock_runtime_client():
        """
        This function will return bedrock runtime client.
        """
        bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            aws_session_token="IQoJb3JpZ2luX2VjEFgaCXVzLWVhc3QtMSJIMEYCIQCyOaIrHCax2JTb+5IJoFGps/mLSK+TG7hWQKPxdIlzrAIhAMIxeBikXE4yfBhQdDhFQH0dWi6A/aO3mnSomUi9mtFJKo8DCKH//////////wEQABoMNDcxMTEyOTY3MjIwIgzLLy2pkiU+o5M8Q3Qq4wLWTX34XQCsCiOc1Z69fCuAg0Pu0nZvpxSY9uVRBFeGqC+oqoJIUS3YPNDbGqrhOHB/6xLfXalVMl7gQc5nHcZrQEjhUGhn/9/ZSAcP7wKQdLZqjRtbqdnB3GteaxSUaH77OIabpJmwb9MMf3socZdxeBBmgXCXcid6sG5MXOI8l4CeC4XwHZqb1nVcEhsoVgXem96nWmwHuAHIEeBg4HN2kwl/pnQE6LhGoik43uHPsLOw82XDE9ABZ4HIYxtJNZLI6MsR9nxM69Q2Nl3iqg5WqYPoP1za45AfYfjL3UqNn/0BZnIZULcxINuPiH/u/1SBfPfSijF61SvsDUq177vm44FwaDA+jQdHL9aFYZ0YjGUjS/9TKjMqaWGBHa0MjpW2rjLqg0L65+3N7djtn9jNUQpm20r5hJaX1UUoHjwds1WVTFmCZ9ckdUeHDTM29F/svkHIQxrsJwF9zisgWnKqlBaNMKSSqLEGOqUBx19pV4TEdu5jWRduX8gaQu8Y7pb11JJwPQzjl57qyz/LevPZWrxm3xdXiSabaOKzzkXKXLZG2x87OcjmYhk58RVLmJ6zaEq2cNfJBFaiJkOHMvkEWhZ4jLh7SLal4/ZZybTyw0jjqMUR3ui8JQUt5MmY9Ms7p1CBJvIcCUY7Ea8cV2Xt9RdVqfJVj9ort0NyKghKU6lDlHOVrP5owgHfJEIiH9tT"

        )

        return bedrock_runtime_client
