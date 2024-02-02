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
            aws_secret_access_key=SECRET_ACCESS_KEY
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
            aws_secret_access_key=SECRET_ACCESS_KEY
        )

        return bedrock_runtime_client

