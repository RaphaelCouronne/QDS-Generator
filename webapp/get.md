# Web server

Edit this page for the default text posted on the website.


## How to use

### Install requirements
```bash
pip install -r requirements.txt
```

#### Optional
Also consider installing [httpie](http://httpie.org) for simpler-than-curl command-line requests

### Run server
```bash
python web_server.py
```

### Make requests on the app

(Use another terminal)

#### View the app - `GET` Request

Open your browser on [localhost:8080](http://localhost:8080/) or do a http request:
```bash
curl -X GET http://localhost:8080/
http GET http://localhost:8080/
```

You should see this readme file.

#### Make predictions - `POST` request

Prepare your data in a JSON file with at least a `"data"` field (see [payload.json](payload.json) for an example):
```json
{
  "data": "your data here"
}
```


Then send it to the app:

```bash
# With curl
curl -X POST -H "Content-Type: application/json" http://localhost:8080 -d @payload.json

# Simpler with httpie
http POST http://localhost:8080 < payload.json
```
