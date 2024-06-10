# Third-party imports
from fastapi import FastAPI, Form, Depends, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse
from decouple import config

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = config("TWILIO_ACCOUNT_SID")
auth_token = config("TWILIO_AUTH_TOKEN")
openai_api_key = config("OPENAI_API_KEY")
client = Client(account_sid, auth_token)
twilio_number = config('TWILIO_NUMBER')

# Internal imports
from models import Conversation, SessionLocal
from utils import logger, run_rag_query
import logging

app = FastAPI()

# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

@app.post("/message")
async def reply(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    logger.info('Senging WhatsApp Mesage')
    # Extract the phone number from the incoming webhook request
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    print(f"Sending the LangChain response to this number: {whatsapp_number}")

    # Get the generated text from the LangChain agent
    langchain_response = run_rag_query(Body)
    
    # Store the conversation in the database
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=langchain_response
            )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in database: {e}")
    
    # Now send the message
    try:
        response = MessagingResponse()
        msg = response.message()
        msg.body(langchain_response)
        xml_response = str(response)
        logging.info(f"Outgoing response: {xml_response}")
        return PlainTextResponse(xml_response, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error sending message to {whatsapp_number}: {e}")
        return PlainTextResponse("Error processing request", status_code=500)

