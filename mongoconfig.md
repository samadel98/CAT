
## Deploy mongodb in as a Docker container

First you need to install [Docker](https://docs.docker.com/install/) in the host, then use the following steps:

### Get the image

```bash
docker pull mongo
```

### Run the container
```bash
docker run -d -p 27017-27019:27017-27019 -v <path/db/local/host>:/data/db --name mongodb -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=strong_password  mongo
```

### Enter to the container
```bash
docker exec -it mongodb bash
```

### Conect with the admin user
```bash
mongosh -u admin -p <previous password>
```

### Exit the mongo shell and the container
```bash
exit
exit
```
