"use strict";

// A tiny in-memory HTTP/HTTPS proxy.  It is launched by WSL through Windows
// Node.js so outbound sockets follow the Windows VPN/TUN route.  The server is
// bound only to WSL's host-gateway interface and writes no Windows files.
const http = require("http");
const net = require("net");

const listenHost = process.argv[2];
const listenPort = Number(process.argv[3] || "5780");
if (!listenHost || !Number.isInteger(listenPort) || listenPort < 1 || listenPort > 65535) {
  throw new Error("usage: node windows-connect-proxy.cjs HOST [PORT]");
}

const server = http.createServer((request, response) => {
  let target;
  try {
    target = new URL(request.url);
  } catch {
    response.writeHead(400);
    response.end();
    return;
  }
  if (target.protocol !== "http:") {
    response.writeHead(400);
    response.end();
    return;
  }

  const headers = { ...request.headers, host: target.host };
  delete headers["proxy-connection"];
  delete headers["proxy-authorization"];
  const upstream = http.request(
    {
      hostname: target.hostname,
      port: target.port || 80,
      path: target.pathname + target.search,
      method: request.method,
      headers,
    },
    (upstreamResponse) => {
      response.writeHead(upstreamResponse.statusCode || 502, upstreamResponse.headers);
      upstreamResponse.pipe(response);
    },
  );
  upstream.on("error", () => {
    if (!response.headersSent) response.writeHead(502);
    response.end();
  });
  request.pipe(upstream);
});

server.on("connect", (request, client, head) => {
  let target;
  try {
    target = new URL(`http://${request.url}`);
  } catch {
    client.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    return;
  }

  let connected = false;
  const upstream = net.connect(Number(target.port) || 443, target.hostname, () => {
    connected = true;
    client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    if (head.length) upstream.write(head);
    client.pipe(upstream);
    upstream.pipe(client);
  });
  upstream.on("error", () => {
    if (!connected && !client.destroyed) {
      client.end("HTTP/1.1 502 Bad Gateway\r\n\r\n");
    } else {
      client.destroy();
    }
  });
  client.on("error", () => upstream.destroy());
});

server.on("clientError", (_error, socket) => {
  if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
});

server.on("error", (error) => {
  process.stderr.write(`${error.stack || String(error)}\n`);
  process.exit(1);
});

server.listen(listenPort, listenHost, () => {
  process.stdout.write(`READY ${listenHost}:${listenPort}\n`);
});
