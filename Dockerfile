# TouchTerrain_for_MPH server / runner image
#   build:  docker compose build
#   run a multicolor job:  docker compose run --rm touchterrain python TouchTerrain_multicolor.py /jobs/<job>.json
#   web app:  docker compose up -d   (port set in docker-compose.yml)
FROM mambaorg/micromamba:1.5.10

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /app/requirements.txt
# everything else from requirements.txt via pip; GDAL and numpy already come from conda
RUN grep -viE "^(gdal|numpy)" requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt
# multicolor: exact inlay pockets (not on conda-forge). Its wheel links the C++
# runtime, so point the loader at conda's newer copy (GDAL needs that one).
RUN pip install --no-cache-dir "manifold3d>=3.5,<4"

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
# the container may run as your own user id (see docker-compose.yml), so let it write here
USER root
RUN chmod -R a+rwX /app
USER $MAMBA_USER

ENV PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/conda/lib \
    PYTHONUNBUFFERED=1 \
    TOUCHTERRAIN_EE_SA_CREDS=/secrets/ee-key.json \
    TOUCHTERRAIN_TMP_FOLDER=/output/server/tmp \
    TOUCHTERRAIN_DOWNLOADS_FOLDER=/output/server/downloads \
    TOUCHTERRAIN_PREVIEWS_FOLDER=/output/server/previews

EXPOSE 8080
CMD ["gunicorn", "-c", "touchterrain/server/gunicorn_settings.py"]
