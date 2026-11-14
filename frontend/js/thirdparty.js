// Third-party provider acknowledgement helpers (modal naming the provider host).
import { api } from "./api.js";
import { h } from "./dom.js";
import { modal } from "./ui.js";
import { hostOf } from "./render_model.mjs";

/** Modal naming the host. Resolves true only on explicit confirmation. */
export function confirmThirdParty({ host, alias, what }) {
  const body = h("div", null,
    h("p", null, "服务商 ", h("strong", { text: alias || "（未命名）" }), " 的地址为 ", h("strong", { class: "mono", text: host || "（未知主机）" }), "，不是本机地址，属于第三方服务。"),
    h("p", { text: what || "作文内容（文字或照片）会被发送到该主机处理。请确认你有权这样做，并已知晓相关隐私风险。" }));
  return modal({ title: "将数据发送给第三方服务", body, buttons: [{ label: "取消", value: false }, { label: "我已知晓，继续", primary: true, value: true }] }).then((v) => v === true);
}

/** Show the modal for a provider object ({id, alias, base_url}) using an already known status; no network. */
export const confirmForProvider = (p, what) => confirmThirdParty({ host: hostOf(p && p.base_url), alias: p && p.alias, what });

/**
 * Pre-flight: ask the server whether this provider needs the user's acknowledgement, and if so ask the user.
 * Resolves {ok:boolean, ack:boolean}: ok=false when the user declined; ack=true means send third_party_ack:true.
 * A failing ack-status request is treated as "unknown": proceed without flag (the server answers 409 if it is needed).
 */
export async function preflightAck(provider, what) {
  let st = null;
  try { st = await api.ackStatus(provider.id); } catch { return { ok: true, ack: false }; }
  if (st && st.is_third_party && !st.acked) {
    const yes = await confirmThirdParty({ host: st.host || hostOf(provider.base_url), alias: provider.alias, what });
    return yes ? { ok: true, ack: true } : { ok: false, ack: false };
  }
  return { ok: true, ack: false };
}
