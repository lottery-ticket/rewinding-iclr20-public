FROM nvidia/cuda:10.1-runtime

        #################### common base configuration ####################

RUN apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        locales                     \
        build-essential             \
        cmake                       \
        coreutils                   \
        curl                        \
        git                         \
        gnupg                       \
        man                         \
        python                      \
        qt5-default                 \
        sudo                        \
        tmux                        \
        vim                         \
        wget

# install mysql
RUN wget -O /tmp/RPM-GPG-KEY-mysql https://repo.mysql.com/RPM-GPG-KEY-mysql && \
        apt-key add /tmp/RPM-GPG-KEY-mysql && \
        /bin/echo -e "deb http://repo.mysql.com/apt/debian/ stretch mysql-5.7\ndeb-src http://repo.mysql.com/apt/debian/ stretch mysql-5.7" > /etc/apt/sources.list.d/mysql.list && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

COPY misc/tmux_attach.sh /etc/tmux_attach.sh
RUN chmod 555 /etc/tmux_attach.sh

# set locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG="en_US.UTF-8" LANGUAGE="en_US:en" LC_ALL="en_US.UTF-8"

        #################### custom base configuration ####################

ARG ROOT_INSTALL_FILE
COPY "${ROOT_INSTALL_FILE:-misc/noop.sh}" /tmp/_root_install.sh
RUN bash /tmp/_root_install.sh

        #################### create and configure user ####################

ARG PROJECT_NAME
RUN echo "${PROJECT_NAME}" | grep -q '^[a-z][-a-z0-9_]*$' # check that PROJECT_NAME is legal
        # unfortunately we can't run nontrivial commands on the lhs of an ENV, so get the uppercase project name passed too
ARG PROJECT_NAME_UPPERCASE
RUN [ "$(echo ${PROJECT_NAME} | awk '{print toupper($0)}')" = "${PROJECT_NAME_UPPERCASE}" ] # PROJECT_NAME uppercase === PROJECT_NAME_UPPERCASE

RUN groupadd -g 1000 "${PROJECT_NAME}" && useradd -m -s /bin/bash -r -u 1000 -g "${PROJECT_NAME}" "${PROJECT_NAME}"
USER "${PROJECT_NAME}"
WORKDIR "/home/${PROJECT_NAME}"
ENV "${PROJECT_NAME_UPPERCASE}_HOME"="/home/${PROJECT_NAME}/${PROJECT_NAME}"

        #################### custom user configuration ####################

ARG USER_INSTALL_FILE
COPY --chown=1000:1000 "${USER_INSTALL_FILE:-misc/noop.sh}" /tmp/_user_install.sh
RUN bash /tmp/_user_install.sh

        #################### misc user configuration ####################

COPY --chown=1000:1000 misc/install_misc.sh /tmp/_misc_install.sh
RUN bash -l /tmp/_misc_install.sh

        #################### configure jupyter notebook ###################

COPY --chown=1000:1000 misc/install_notebook.sh /tmp/_notebook_install.sh
RUN bash -l /tmp/_notebook_install.sh
EXPOSE 8888 6006

        #################### set UID and GID correctly ####################

ARG UID=1000
ARG GID=1000

USER root
RUN usermod -u "${UID}" "${PROJECT_NAME}" 2>/dev/null && groupmod -g "${GID}" "${PROJECT_NAME}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/compat"

        ########################## entrypoint ############################

USER "${PROJECT_NAME}"
ENV LOCAL_DOCKER=
