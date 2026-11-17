// Tuned defaults for knobs whose names are built at run time (DUR_CS_P_<ph>, LINE_CAV_<n>_*, LIQ_<x>_*, RED_*, ...),
// so tools/apply_params.py cannot write them next to their param() call. param() (params.h) uses a value here
// in place of the call's default; a params file or the environment still overrides it. Written by the knob search
// (ref/opt/ksearch/LOG.md): run1 variant B, adopted by the owner 2026-11-17.
#ifndef PUHUJA_TUNED_DEFAULTS_H_
#define PUHUJA_TUNED_DEFAULTS_H_

#include <map>
#include <string>

inline const std::map<std::string, float>& tunedDefaults() {
  static const std::map<std::string, float> m = {
    {"DUR_CS_P_AE", 1.00064433f},
    {"DUR_CS_P_AO", 1.0003866f},
    {"DUR_CS_P_G", 1.18184772f},
    {"DUR_CS_P_HH", 1.00167526f},
    {"DUR_CS_P_IH", 1.0007732f},
    {"DUR_CS_P_K", 1.0003866f},
    {"DUR_CS_P_OW", 1.00167526f},
    {"DUR_CS_P_OY", 0.999871134f},
    {"DUR_CS_P_UW", 1.00115979f},
    {"DUR_CS_P_Y", 1.00090206f},
    {"DUR_CS_P_ZH", 0.999742268f},
    {"LINE_AG_FRIC", 0.30038661f},
    {"LINE_CAV_0_FL", -0.0622693309f},
    {"LINE_CAV_10_FA", 0.486494842f},
    {"LINE_CAV_10_FL", 0.618739678f},
    {"LINE_CAV_11_FA", -0.158144326f},
    {"LINE_CAV_11_FL", -0.586005159f},
    {"LINE_CAV_11_HA", 0.210907214f},
    {"LINE_CAV_12_FA", -0.416922693f},
    {"LINE_CAV_12_FL", -0.704350519f},
    {"LINE_CAV_12_HA", -0.592278341f},
    {"LINE_CAV_13_FA", 0.0861108276f},
    {"LINE_CAV_13_FL", -0.0414046386f},
    {"LINE_CAV_13_HA", -0.205092777f},
    {"LINE_CAV_14_FA", -0.197974228f},
    {"LINE_CAV_14_FL", -0.0179536074f},
    {"LINE_CAV_14_HL", 0.853487873f},
    {"LINE_CAV_16_FA", -0.167069582f},
    {"LINE_CAV_16_FL", -0.844206171f},
    {"LINE_CAV_16_HA", -0.410866972f},
    {"LINE_CAV_1_FL", 0.176984543f},
    {"LINE_CAV_1_HA", -0.0595360812f},
    {"LINE_CAV_2_FL", 1.15291233f},
    {"LINE_CAV_2_HL", -0.277894322f},
    {"LINE_CAV_3_FA", -0.169653351f},
    {"LINE_CAV_3_FL", -0.679070854f},
    {"LINE_CAV_3_HA", -0.6423402f},
    {"LINE_CAV_3_HL", -0.130469071f},
    {"LINE_CAV_4_FA", -0.621069583f},
    {"LINE_CAV_4_HA", -0.293737105f},
    {"LINE_CAV_6_FA", -0.247680408f},
    {"LINE_CAV_6_FL", -0.434329895f},
    {"LINE_CAV_6_HA", 0.673211321f},
    {"LINE_CAV_6_HL", -0.0465760293f},
    {"LINE_CAV_8_FA", -0.116298966f},
    {"LINE_CAV_8_HA", -0.720757758f},
    {"LINE_CAV_8_HL", 0.134881438f},
    {"LINE_CAV_9_FA", 0.118386601f},
    {"LINE_CAV_9_FL", 0.450123711f},
    {"LINE_CAV_9_HA", 0.646136618f},
    {"LINE_ROOM_TONE", 0.0376572153f},
    {"LIQ_J_APH", 1.00773196f},
    {"LIQ_J_LIPL", 0.992268041f},
    {"LIQ_LC_APH", 1.0128866f},
    {"LIQ_LC_LIPA", 0.996134021f},
    {"LIQ_LC_LIPL", 1.00515464f},
    {"LIQ_L_APH", 0.99871134f},
    {"LIQ_L_LIPA", 1.01159794f},
    {"LIQ_L_LIPL", 0.988402062f},
    {"LIQ_RC_APH", 0.605412395f},
    {"LIQ_RC_LIPA", 1.00257732f},
    {"LIQ_RC_LIPL", 1.00773196f},
    {"LIQ_R_LIPL", 0.992268041f},
    {"LIQ_W_APH", 0.992268041f},
    {"LIQ_W_LIPA", 1.01675258f},
    {"LIQ_W_LIPL", 1.0128866f},
    {"LIQ_W_ROW", 11.5572805f},
    {"RED_I_N_LAB", 0.599226828f},
    {"RED_I_N_PAL", 1.00386598f},
    {"RED_I_N_VEL", 1.01159794f},
    {"RED_I_P_L", 1.02263292f},
    {"RED_I_P_R", 0.808247435f},
    {"RED_I_P_W", 0.50064433f},
    {"RED_S_N_COR", 0.451159782f},
    {"RED_S_N_L", 0.99742268f},
    {"RED_S_N_PAL", 0.345489685f},
    {"RED_S_N_V", 1.00773196f},
    {"RED_S_N_VEL", 0.0988402072f},
    {"RED_S_P_COR", 0.297293826f},
    {"RED_S_P_R", 0.0989690732f},
    {"RED_S_P_V", 0.492912371f},
    {"RED_S_P_W", 0.603092808f},
    {"DUR_CS_FUNC@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"DUR_CS_H@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_ACC_WORD_LEN@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_LONG_WORD_LEN@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_UNACC_CONS_LEN@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_UNACC_WORD_LEN@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_IH_LEN@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"EN_PRE_FINAL_T_VOWEL@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"TRANS_G_W@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"TRANS_V_W@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"TRANS_PAUSE_REL_W@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"LINE_ANTICIP_M_SCALE@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"LINE_ANTICIP_N_SCALE@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"LINE_EH_W@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"LINE_SCHWA_W@f", 1.0f},  // VOWDICT agent36 #2: female only, the pre-search value
    {"LINE_TR_D_PRESS@f", 1.0f},  // ONSET agent36 #1: female word-initial dɹ, CL agent20's fricated alveolar slit
    {"LINE_TR_D_SLIT_X@f", 0.83f},
    {"LINE_TR_D_FRIC@f", 3.0f},
  };
  return m;
}

#endif  // PUHUJA_TUNED_DEFAULTS_H_
