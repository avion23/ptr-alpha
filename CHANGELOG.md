# Changelog

## Unreleased

### Breaking changes

- `score_ticker_by_buyers` now defaults to consensus scoring, requires an explicit
  `as_of_date` in that mode, and no longer accepts the unused member-skill,
  uncertainty-penalty, or solo-buyer posterior-gate parameters.
- Removed `_lookup_buyer_posterior_lift` from the member-ranking package API.
- `MemberSkillPosterior` now exposes only estimable member effects and effective
  information. `score_members_for_ticker` was replaced by
  `score_member_posteriors`, which accepts unique member identities only.
- `bayesian_quality` scoring was removed. Unknown scoring modes now fail instead
  of silently falling back to another score.
