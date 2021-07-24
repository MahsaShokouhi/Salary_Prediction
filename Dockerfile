FROM python:3.8

# RUN pip install virtualenv
# ENV VIRTUAL_ENV=/venv
# RUN virtualenv venv -p python3
# ENV PATH="VIRTUAL_ENV/bin:$PATH"

COPY ./app /app
COPY ./model /model
WORKDIR /app


# Install dependencies
RUN pip install -r requirements.txt

# Expose port 
EXPOSE 8080

# Run the application:
CMD ["python", "app.py"]
