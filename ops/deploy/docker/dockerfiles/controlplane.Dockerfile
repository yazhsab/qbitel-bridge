# syntax=docker/dockerfile:1
FROM golang:1.22 AS build
WORKDIR /src
COPY go/controlplane /src
RUN cd /src && go build -o /out/controlplane ./cmd/controlplane

FROM gcr.io/distroless/base-debian12:nonroot
COPY --from=build /out/controlplane /controlplane
USER nonroot
ENTRYPOINT ["/controlplane"]
