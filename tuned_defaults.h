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
    {"LINE_AG_FRIC", 0.314004579f},
    {"LINE_CAV_0_FL", -0.06297732f},
    {"LINE_CAV_10_FA", 0.466760491f},
    {"LINE_CAV_10_FL", 0.657520718f},
    {"LINE_CAV_11_FA", -0.166973007f},
    {"LINE_CAV_11_FL", -0.5827714f},
    {"LINE_CAV_11_HA", 0.214639121f},
    {"LINE_CAV_12_FA", -0.416063333f},
    {"LINE_CAV_12_FL", -0.71321f},
    {"LINE_CAV_12_HA", -0.60543746f},
    {"LINE_CAV_13_FA", 0.0866993489f},
    {"LINE_CAV_13_FL", -0.0392321437f},
    {"LINE_CAV_13_HA", -0.20134969f},
    {"LINE_CAV_14_FA", -0.203772389f},
    {"LINE_CAV_14_FL", -0.0179536074f},
    {"LINE_CAV_14_HL", 0.872392214f},
    {"LINE_CAV_16_FA", -0.1660575f},
    {"LINE_CAV_16_FL", -0.8396854f},
    {"LINE_CAV_16_HA", -0.44856977f},
    {"LINE_CAV_1_FL", 0.170468038f},
    {"LINE_CAV_1_HA", -0.0621364918f},
    {"LINE_CAV_2_FL", 1.1130055f},
    {"LINE_CAV_2_HL", -0.289246353f},
    {"LINE_CAV_3_FA", -0.169653351f},
    {"LINE_CAV_3_FL", -0.664475765f},
    {"LINE_CAV_3_HA", -0.639014331f},
    {"LINE_CAV_3_HL", -0.12432881f},
    {"LINE_CAV_4_FA", -0.60531899f},
    {"LINE_CAV_4_HA", -0.302016192f},
    {"LINE_CAV_6_FA", -0.247680408f},
    {"LINE_CAV_6_FL", -0.434329895f},
    {"LINE_CAV_6_HA", 0.6764734f},
    {"LINE_CAV_6_HL", -0.0465760293f},
    {"LINE_CAV_8_FA", -0.118804364f},
    {"LINE_CAV_8_HA", -0.72061518f},
    {"LINE_CAV_8_HL", 0.134713932f},
    {"LINE_CAV_9_FA", 0.118091899f},
    {"LINE_CAV_9_FL", 0.45569025f},
    {"LINE_CAV_9_HA", 0.6408806f},
    {"LINE_ROOM_TONE", 0.0376572153f},
    {"LIQ_J_APH", 1.0046613f},
    {"LIQ_J_LIPL", 0.992268041f},
    {"LIQ_LC_APH", 0.980244948f},
    {"LIQ_LC_LIPA", 0.98758375f},
    {"LIQ_LC_LIPL", 1.0065812f},
    {"LIQ_L_APH", 0.99871134f},
    {"LIQ_L_LIPA", 1.01159794f},
    {"LIQ_L_LIPL", 0.948792877f},
    {"LIQ_RC_APH", 0.6089756f},
    {"LIQ_RC_LIPA", 1.0014272f},
    {"LIQ_RC_LIPL", 1.00773196f},
    {"LIQ_R_LIPL", 1.04697504f},
    {"LIQ_W_APH", 0.992268041f},
    {"LIQ_W_LIPA", 1.01675258f},
    {"LIQ_W_LIPL", 1.0238227f},
    {"LIQ_W_ROW", 11.5572805f},
    {"RED_I_N_LAB", 0.599226828f},
    {"RED_I_N_PAL", 1.00386598f},
    {"RED_I_N_VEL", 1.01159794f},
    {"RED_I_P_L", 0.973840917f},
    {"RED_I_P_R", 0.808247435f},
    {"RED_I_P_W", 0.50064433f},
    {"RED_S_N_COR", 0.451159782f},
    {"RED_S_N_L", 0.942558033f},
    {"RED_S_N_PAL", 0.349212502f},
    {"RED_S_N_V", 1.02032856f},
    {"RED_S_N_VEL", 0.0971154573f},
    {"RED_S_P_COR", 0.297293826f},
    {"RED_S_P_R", 0.097312546f},
    {"RED_S_P_V", 0.493677536f},
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
    {"BREATH_PUNCT", 0.299971446f},  // run8
    {"LINE_CAV_0_FA", 0.157638865f},  // run8
    {"LIQ_R_APH", 0.588596656f},  // run11
    {"RED_S_N_LAB", 0.09734539f},  // run122
    {"LIQ_R_LIPA", 0.991868917f},  // run65
    {"RED_I_N_L", 0.967260062f},  // run60
    {"LINE_CAV_13_HL", 0.499041781f},  // run14
    {"LINE_CAV_12_HL", -0.296380587f},  // run8
    {"LINE_CAV_14_HA", -0.0261667f},  // run8
    {"LINE_CAV_1_HL", 0.35288771f},  // run8
    {"LINE_CAV_8_FL", -0.19120417f},  // run8
    {"LIQ_J_LIPA", 1.01615672f},  // run8
    {"LIQ_W_MIX", 0.49846286f},  // run8
    {"LINE_CAV_0_HA", 0.432249426f},  // run7
    {"LINE_CAV_10_HA", 0.6265658f},  // run7
    {"LINE_CAV_10_HL", 0.0936515866f},  // run7
    {"LINE_CAV_11_HL", 0.0329763436f},  // run7
    {"LINE_CAV_1_FA", -0.28113738f},  // run7
    {"LINE_CAV_2_HA", -0.749368f},  // run7
    {"LINE_CAV_4_FL", -0.77026794f},  // run7
    {"LINE_CAV_9_HL", -0.11514974f},  // run7
  };
  return m;
}

#endif  // PUHUJA_TUNED_DEFAULTS_H_
