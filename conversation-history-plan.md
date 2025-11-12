# Conversation History Plan

1. ~~**Persist Conversations**~~ ✅
   - Add a lightweight storage layer (e.g., `nanochat/storage.py`) that wraps `sqlite3` with tables for `conversations(id, title, created_at, updated_at)` and `messages(id, conversation_id, role, content, created_at)`.
   - Initialize the database during FastAPI startup inside `lifespan` (`scripts/chat_web.py:223`) and attach the store to `app.state` for reuse in endpoints.
   - Create helper methods to create a conversation, append messages, list summaries, fetch full threads, rename, and soft-delete so later UX enhancements are easy.

2. ~~**Extend Existing Chat API**~~ ✅
   - Update `ChatRequest` in `scripts/chat_web.py:139` to accept an optional `conversation_id`.
   - When `/chat/completions` (`scripts/chat_web.py:313`) receives a request: if `conversation_id` is missing, create a new conversation before token generation; otherwise load history, append incoming user message, and ensure the message log passed to the model matches what’s stored.
   - After streaming ends, persist the assistant reply and emit the active `conversation_id` (and maybe a refreshed `title`) in the final SSE chunk so the UI can keep track without restarting the stream.
   - Guard rails: return 404 if a bad `conversation_id` is provided; cap total stored tokens to stay under the existing abuse limits.

3. **Conversation Management Endpoints**
   - Introduce Pydantic schemas for summaries and full conversations.
   - Add routes in `scripts/chat_web.py`:
     - `GET /conversations` → list recent conversations with `id`, preview text, timestamps.
     - `GET /conversations/{id}` → return ordered messages for hydration when the user resumes a thread.
     - (Optional but easy) `DELETE /conversations/{id}` or `PATCH /conversations/{id}` to rename, giving users control over their list.
   - These endpoints simply call the storage helpers and return JSON; no model invocation required.

4. **Restructure the UI Layout**
   - Update `nanochat/ui.html:250` to add a left sidebar for the history list plus a “New Chat” button, with the existing chat window occupying the right pane.
   - Adjust the CSS to support the two-column layout and provide visual feedback for the selected conversation, along with responsive behavior for narrow screens (sidebar collapses into a drawer/button).

5. **Front-End State & Networking**
   - ~~On load, fetch `/conversations` and render the list; clicking an item fetches `/conversations/{id}`, populates `messages`, and re-renders the thread.~~ ✅
   - ~~Track `activeConversationId` in the JS block starting at `nanochat/ui.html:275`; include it whenever calling `/chat/completions`.~~ ✅
   - ~~When the user starts a brand-new chat, call a small helper endpoint (or rely on the SSE “new conversation” signal) to acquire the new `conversation_id`, reset the message pane, and focus the input.~~ ✅
   - ~~After each assistant response, refresh the sidebar entry (last message preview + timestamp) so the history keeps itself up to date.~~ ✅

6. **Validation & QA**
   - Add simple unit tests for the storage layer (e.g., using `pytest` to create an in-memory DB) that cover creating, listing, and resuming conversations.
   - Manually test in the browser: start multiple chats, reload the page to confirm persistence, resume a conversation, send additional prompts, and ensure the right thread receives the continuation.
   - Verify regressions: SSE stream still works for long outputs, abuse limits remain enforced, and shortcuts like Ctrl+Shift+N still spawn a fresh conversation tied to the backend.
