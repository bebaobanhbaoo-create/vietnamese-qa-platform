# Airflow with the project installed in its own virtualenv: the project pins its ML stack
# (pydantic, fastapi, torch, ...) and would conflict with Airflow's dependencies if mixed.
FROM apache/airflow:3.3.2-python3.11

USER root
RUN python -m venv /opt/huflit-venv && chown -R airflow: /opt/huflit-venv
USER airflow

RUN /opt/huflit-venv/bin/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=airflow pyproject.toml README.md /opt/huflit-src/
COPY --chown=airflow src /opt/huflit-src/src
RUN /opt/huflit-venv/bin/pip install --no-cache-dir /opt/huflit-src

# The repository (data/, eval/, docs/) is mounted at /opt/huflit at runtime.
ENV HUFLIT_BIN=/opt/huflit-venv/bin/huflit-qa \
    HUFLIT_ROOT_DIR=/opt/huflit
