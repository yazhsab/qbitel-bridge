# syntax=docker/dockerfile:1
FROM golang:1.22 AS build
WORKDIR /src
COPY go/mgmtapi /src
RUN cd /src && go build -o /out/mgmtapi ./cmd/mgmtapi

FROM gcr.io/distroless/base-debian12:nonroot
COPY --from=build /out/mgmtapi /mgmtapi
USER nonroot
ENTRYPOINT ["/mgmtapi"]
