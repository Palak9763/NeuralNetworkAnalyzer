/**
 * api/client.ts
 *
 * Why this file exists:
 *   Single place that knows the backend's base URL and endpoint shapes.
 *   Components never call axios/fetch directly - they call these
 *   functions, so if the API contract changes, only this file needs edits.
 *
 * How it connects:
 *   Used by components/UploadPage.tsx (uploadProject) and
 *   components/GraphCanvas.tsx (fetchGraph, via polling).
 */

import axios from "axios";
import type { SourceResponse, UniversalGraph, UploadResponse } from "../types/graph";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function uploadProject(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await axios.post<UploadResponse>(`${API_BASE}/api/v1/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function fetchGraph(jobId: string): Promise<UniversalGraph> {
  const response = await axios.get<UniversalGraph>(`${API_BASE}/api/v1/graph/${jobId}`);
  return response.data;
}

export async function fetchUploads(): Promise<Array<{job_id: string; filename: string; uploaded_at: string;}>> {
  const response = await axios.get(`${API_BASE}/api/v1/uploads`);
  return response.data;
}

export async function fetchSource(jobId: string): Promise<SourceResponse> {
  const response = await axios.get<SourceResponse>(`${API_BASE}/api/v1/source/${jobId}`);
  return response.data;
}

/* ── Auth endpoints ─────────────────────────────────── */

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  user_id: string;
  email: string;
  is_active: boolean;
}

export async function loginUser(email: string, password: string): Promise<AuthResponse> {
  const response = await axios.post<AuthResponse>(`${API_BASE}/api/v1/auth/login`, { email, password });
  return response.data;
}

export async function registerUser(email: string, password: string): Promise<UserResponse> {
  const response = await axios.post<UserResponse>(`${API_BASE}/api/v1/auth/register`, { email, password });
  return response.data;
}

