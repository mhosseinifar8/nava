//! Ultra-Low-Latency Axum WebSocket Streaming Gateway for Nava.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;

use nava_core_rs::{clean_for_matcha, normalize_persian, split_into_streaming_clauses};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    gateway: &'static str,
}

#[derive(Deserialize)]
struct NormalizeRequest {
    text: String,
    mode: Option<String>,
}

#[derive(Serialize)]
struct NormalizeResponse {
    normalized: String,
    clauses: Vec<String>,
}

async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        gateway: "rust-axum-core",
    })
}

async fn normalize_handler(Json(payload): Json<NormalizeRequest>) -> impl IntoResponse {
    let mode = payload.mode.as_deref().unwrap_or("general");
    let norm = if mode == "matcha" {
        clean_for_matcha(&payload.text)
    } else {
        normalize_persian(&payload.text)
    };
    let clauses = split_into_streaming_clauses(&norm);
    Json(NormalizeResponse {
        normalized: norm,
        clauses,
    })
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(msg_result) = socket.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            Message::Text(txt) => {
                // Check for ping frame
                if txt.contains(r#""type":"ping""#) || txt.contains(r#""type": "ping""#) {
                    let pong = serde_json::json!({
                        "type": "pong",
                        "status": "alive"
                    });
                    if socket.send(Message::Text(pong.to_string())).await.is_err() {
                        break;
                    }
                    continue;
                }

                // Parse incoming request
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                        let cleaned = clean_for_matcha(text);
                        let clauses = split_into_streaming_clauses(&cleaned);
                        let resp = serde_json::json!({
                            "type": "clauses",
                            "clauses": clauses,
                            "count": clauses.len()
                        });
                        let _ = socket.send(Message::Text(resp.to_string())).await;
                    }
                }
            }
            Message::Ping(payload) => {
                if socket.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/text/normalize", post(normalize_handler))
        .route("/ws/stream", get(ws_handler))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], 8090));
    println!("Nava Rust Fast-Path Gateway listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

