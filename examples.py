from data import *
# start English z suffix devoicing examples
consonant = u_seg(['k', 'd', 'h', 't', 'ʔ', 'w'])
voiceless_consonant = u_seg(['k', 'h', 't', 'ʔ'])
vowel = u_seg(['æ', 'ɔ', 'ɛ', 'e', 'I'])
# TODO consider making these proportions random variables
s_z_seg = PossibleSegments([('s', 0.8), ('z', 0.2)])
f_v_seg = PossibleSegments([('f', 0.8), ('v', 0.2)])

cat_sz = [consonant, vowel, seg('t'), s_z_seg]
gz = [consonant, vowel, seg('g'), seg('z')]
nz = [consonant, vowel, seg('n'), seg('z')]
five_f_v_th = [
    voiceless_consonant,
    voiceless_consonant,
    vowel,
    seg('l'),
    f_v_seg,
    seg('θ')]
four_f_v_th = [consonant, vowel, seg('l'), f_v_seg, seg('θ')]
two_vowel_t_theta = [consonant, vowel, vowel, seg('t'), seg('θ')]
one_vowel_t_theta = [consonant, vowel, seg('t'), seg('θ')]
n_theta = [consonant, vowel, seg('n'), seg('θ')]

end_voi_words = list(map(Word, [
    cat_sz,
    gz,
    nz,
    five_f_v_th,
    four_f_v_th,
    two_vowel_t_theta,
    one_vowel_t_theta,
    n_theta]))

agree, ident_voi, star_d, star_d_sigma = 'Agree', '*Ident-IO(voi)', '*D', '*D_sigma'
english_voi: Ranking = [agree, ident_voi, star_d, star_d_sigma]
non_english_voi: Ranking = [ident_voi, agree, star_d, star_d_sigma]
voi_rankings = PossibleRankings([(english_voi, 0.8), (non_english_voi, 0.2)])

end_voi_examples = [(word, voi_rankings) for word in end_voi_words]
# end English z suffix devoicing examples

# start hypothetical simple
star_top_voi: Ranking = [star_d, star_d_sigma, ident_voi]
voi_voiceless_rankings = PossibleRankings(
    [(english_voi, 0.5), (star_top_voi, 0.5)])

voice_voiceless_obs = PossibleSegments(
    [('k', 0.25), ('g', 0.25), ('t', 0.25), ('d', 0.25)])


cv = [voice_voiceless_obs, vowel]
vc = [vowel, voice_voiceless_obs]
hypo_voi_words = end_voi_words = list(map(Word, [cv, vc]))

hypo_voi_examples = [(word, voi_rankings) for word in hypo_voi_words]
