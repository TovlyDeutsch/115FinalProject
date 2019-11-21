from data import *
from segments import voiced_consonants, voiceless_consonants, consonants, vowels, voice_obstruents, voiceless_obstruents
# start English z suffix devoicing examples
consonant = u_seg(consonants)
voiceless_consonant = u_seg(voiceless_consonants)
voiced_consonant = u_seg(voiced_consonants)
vowel = u_seg(vowels)
# TODO consider making these proportions random variables

cats = [consonant, vowel, seg('t'), seg('s')]
catz = [consonant, vowel, seg('t'), seg('z')]
gz = [consonant, vowel, seg('g'), seg('z')]
nz = [consonant, vowel, seg('n'), seg('z')]
five_v_th = [
    voiceless_consonant,
    voiceless_consonant,
    vowel,
    seg('l'),
    seg('v'),
    seg('θ')]
five_f_th = [
    voiceless_consonant,
    voiceless_consonant,
    vowel,
    seg('l'),
    seg('f'),
    seg('θ')]
four_v_th = [consonant, vowel, seg('l'), seg('v'), seg('θ')]
four_f_th = [consonant, vowel, seg('l'), seg('f'), seg('θ')]
two_vowel_t_theta = [consonant, vowel, vowel, seg('t'), seg('θ')]
one_vowel_t_theta = [consonant, vowel, seg('t'), seg('θ')]
n_theta = [consonant, vowel, seg('n'), seg('θ')]

end_voi_words = list(map(Word, [
    cats,
    catz,
    gz,
    nz,
    five_f_th,
    five_v_th,
    four_f_th,
    four_v_th,
    two_vowel_t_theta,
    one_vowel_t_theta,
    n_theta]))
sos = '<sos>'  # TODO make this an automatic prepend for rankings
eos = '<eos>'  # TODO make this an automatic apend for rankings
agree, ident_voi, star_d, star_d_sigma = 'Agree', '*Ident-IO(voi)', '*D', '*D_sigma'
english_voi: Ranking = [sos, agree, ident_voi, star_d, star_d_sigma, eos]
faith_voi: Ranking = [sos, ident_voi, agree, star_d, star_d_sigma, eos]

end_voi_examples = []
for word in end_voi_words:
  if word.segments[-1].all_obstruent() and word.segments[-2].all_obstruent():
    if word.segments[-1].all_voiced() == word.segments[-2].all_voiced():
      ranking = single_ranking(english_voi)
    else:
      ranking = single_ranking(faith_voi)
  else:
    ranking = single_ranking(faith_voi)
  end_voi_examples.append((word, ranking))

# end English z suffix devoicing examples

# start hypothetical simple examples
faith_voi: Ranking = [sos, ident_voi, agree, star_d, star_d_sigma, eos]
star_top_voi: Ranking = [sos, star_d, agree, ident_voi, star_d_sigma, eos]

voice_obstruent = u_seg(voice_obstruents)
voiceless_obstruent = u_seg(voiceless_obstruents)

cv_voice = Word([voice_obstruent, vowel])
vc_voice = Word([vowel, voice_obstruent])

cv_voiceless = Word([voiceless_obstruent, vowel])
vc_voiceless = Word([vowel, voiceless_obstruent])

hypo_voi_examples = [
    (cv_voice,
     single_ranking(faith_voi)),
    (vc_voice,
        single_ranking(faith_voi)),
    (cv_voiceless,
        single_ranking(star_top_voi)),
    (vc_voiceless,
        single_ranking(star_top_voi))]

agree_above: Ranking = [sos, agree, ident_voi, star_d, star_d_sigma, eos]
star_above: Ranking = [sos, star_d, agree, ident_voi, star_d_sigma, eos]


# UR is voice, voiceless, vowel
# ccv_agree = Word([voice_obstruent, voice_obstruent, vowel])
ccv_dis1 = Word([voice_obstruent, voiceless_obstruent, vowel])
# ccv_dis2 = Word([voiceless_obstruent, voice_obstruent, vowel])
ccv_agree_voiceless = Word([voiceless_obstruent, voiceless_obstruent, vowel])

star_agree_examples = [
    (ccv_dis1,
        single_ranking(faith_voi)),
    (ccv_agree_voiceless,
        single_ranking(star_above))]

# UR is voice, voiceless, vowel, voice
ccvc_dis1 = Word(
    [voice_obstruent, voiceless_obstruent, vowel, voice_obstruent])
# ccv_dis2 = Word([voiceless_obstruent, voice_obstruent, vowel])
ccvc_agree_voiceless = Word(
    [voiceless_obstruent, voiceless_obstruent, vowel, voiceless_obstruent])

star_agree_double_c_examples = [
    (ccv_dis1,
        single_ranking(faith_voi)),
    (ccv_agree_voiceless,
        single_ranking(star_above))]

# UR is vowel, voice, voiceless, vowel
# vccv_agree = Word([vowel, voice_obstruent, voice_obstruent, vowel])
vccv_dis1 = Word([vowel, voice_obstruent, voiceless_obstruent, vowel])
# vccv_dis2 = Word([vowel, voiceless_obstruent, voice_obstruent, vowel])
vccv_agree_voiceless = Word(
    [vowel, voiceless_obstruent, voiceless_obstruent, vowel])

star_agree_double_vowel_examples = [
    (vccv_dis1,
        single_ranking(faith_voi)),
    (vccv_agree_voiceless,
        single_ranking(star_above))]
