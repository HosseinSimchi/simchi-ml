#### Docker Commands

- ###### `sudo docker compose down`
- ###### `sudo docker-compose up --build`
- ###### `docker compose build --no-cache`
  - ###### When You add some new package inside the `requirements.txt`, Run this command
  - ###### Then Run, `docker compose up`
- ###### Build the docker file 
  - ###### `docker build -t update-model`
  - ###### `docker run --rm update-model`