from face_recognition import face_match
import logging
import boto3
import io
from PIL import Image
from decouple import config

# Load AWS credentials from environment variables
aws_access_key_id = config('AWS_ACCESS_KEY_ID')
aws_secret_access_key = config('AWS_SECRET_ACCESS_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def msg_from_sqs_req():
    # Receive the image file name from sqs_req_queue
    sqs_res_queue = boto3.client('sqs', region_name='us-east-1',
                                 aws_access_key_id=aws_access_key_id, 
                                 aws_secret_access_key=aws_secret_access_key) # initialize sqs_req_queue
    req_queue_url = 'https://sqs.us-east-1.amazonaws.com/905418105068/1229729529-req-queue'   # URL of SQS queue
    
    try:
        # Poll the SQS queue for messages
        response = sqs_res_queue.receive_message(
            QueueUrl=req_queue_url,
            MaxNumberOfMessages=1,  # Maximum number of messages to retrieve
            WaitTimeSeconds=10      # Long polling - how long to wait for a message to arrive
        )

        # Check if there are any messages
        if 'Messages' in response:
            # Retrieve the message(s)
            message = response['Messages'][0]
            
            filename = message['Body']
            
            # Delete the message from the queue (if processing is successful)
            sqs_res_queue.delete_message(
                QueueUrl=req_queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
            return filename
        else:
            logger.info("No messages available in the queue.")
            
    except Exception as e:
        logger.error("An error occurred:", e)
            
def get_image_from_s3(filename):
    try:
        bucket_name = '1229729529-in-bucket'
        s3 = boto3.client('s3', region_name='us-east-1',
                          aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        response = s3.get_object(Bucket=bucket_name, Key=filename)
        image_data = response['Body'].read()
        return image_data
    except Exception as e:
        print(f"Error retrieving image with key {filename} from bucket {bucket_name}: {e}")
        return None


def msg_to_sqs_resp(result):
    try:
        # Send the image file name to sqs_req_queue
        sqs_resp_queue = boto3.client('sqs', region_name='us-east-1',
                                      aws_access_key_id=aws_access_key_id, 
                                      aws_secret_access_key=aws_secret_access_key) # initialize sqs_req_queue
        resp_queue_url = 'https://sqs.us-east-1.amazonaws.com/905418105068/1229729529-resp-queue'   # URL of SQS queue
        msg_body = result

        # Sending msg body to queue
        response = sqs_resp_queue.send_message(
            QueueUrl=resp_queue_url,
            MessageBody=msg_body
        )

        # Log successful message
        logger.info("Message sent successfully. MessageId: %s", response['MessageId'])

    except Exception as e:
        # Log error
        logger.error("An error occurred while sending message: %s", str(e))
        
    
def s3_out_bucket_msg_store(image_name, result):
    try:
        bucket_name = '1229729529-out-bucket'
        s3 = boto3.client('s3', region_name='us-east-1',
                          aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        s3.put_object(Bucket=bucket_name, Key=image_name, Body=result)
        logger.info(f"Uploaded {image_name} to bucket {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {image_name} to bucket {bucket_name}: {e}")
        return False


while True:

    # get the message from sqs_req_queue so that we get image file name to classify
    filename = msg_from_sqs_req()

    # get the image from s3-in-bucket
    image_data = get_image_from_s3(filename)
    image = Image.open(io.BytesIO(image_data))


    # Call the model to classify the image
    result = face_match(image, 'data.pt')
    result = result[0]
    result = str(result)
    # logger.info('printing result from model', result)
    image_name = filename.split('.')[0]
    result_msg = image_name + ':' + result


    # send the classification result to sqs_resp_queue 
    msg_to_sqs_resp(result_msg)

    # store the image classification in s3-out-bucket
    s3_out_bucket_msg_store(image_name, result)