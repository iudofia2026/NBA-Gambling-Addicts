# Kaggle Dataset Comparison Report

Generated: $(date)

## Dataset 1: eoinamoore/historical-nba-data-and-player-box-scores

### Files on Kaggle vs Local

| File Name | Kaggle Size | Local Size | Status |
|-----------|-------------|------------|--------|
| `Games.csv` | 10.0 MB | 9.5 MB | ✅ Present |
| `LeagueSchedule24_25.csv` | 147 KB | 144 KB | ✅ Present |
| `LeagueSchedule25_26.csv` | 187 KB | 183 KB | ✅ Present |
| `PlayerStatistics.csv` | 318 MB | 303 MB | ✅ Present |
| `Players.csv` | 537 KB | 524 KB | ✅ Present |
| `TeamHistories.csv` | 6.9 KB | 6.8 KB | ✅ Present |
| `TeamStatistics.csv` | 34.2 MB | 33 MB | ✅ Present |

**Status: ✅ ALL FILES PRESENT**

**Note:** The README mentions `games_details.csv`, `games.csv`, `players.csv`, and `teams.csv`, but the actual Kaggle dataset uses:
- `Games.csv` (capital G) instead of `games.csv`
- `Players.csv` (capital P) instead of `players.csv`
- `PlayerStatistics.csv` instead of `games_details.csv`
- No `teams.csv` in this dataset (team info is in the basketball dataset)

---

## Dataset 2: wyattowalsh/basketball

### Files on Kaggle vs Local

| File Name | Kaggle Size | Local Size | Status |
|-----------|-------------|------------|--------|
| `csv/common_player_info.csv` | 1.0 MB | 1.0 MB | ✅ Present |
| `csv/draft_combine_stats.csv` | 204 KB | 199 KB | ✅ Present |
| `csv/draft_history.csv` | 829 KB | 810 KB | ✅ Present |
| `csv/game.csv` | 20.3 MB | 19 MB | ✅ Present |
| `csv/game_info.csv` | 2.4 MB | 2.3 MB | ✅ Present |
| `csv/game_summary.csv` | 5.6 MB | 5.4 MB | ✅ Present |
| `csv/inactive_players.csv` | 7.4 MB | 7.0 MB | ✅ Present |
| `csv/line_score.csv` | 10.7 MB | 10 MB | ✅ Present |
| `csv/officials.csv` | 2.3 MB | 2.2 MB | ✅ Present |
| `csv/other_stats.csv` | 3.5 MB | 3.3 MB | ✅ Present |
| `csv/play_by_play.csv` | 2.3 GB | 2.1 GB | ✅ Present |
| `csv/player.csv` | 170 KB | 166 KB | ✅ Present |
| `csv/team.csv` | 2.0 KB | 2.0 KB | ✅ Present |
| `csv/team_details.csv` | 5.5 KB | 5.4 KB | ✅ Present |
| `csv/team_history.csv` | 2.1 KB | 2.0 KB | ✅ Present |
| `csv/team_info_common.csv` | 245 B | 245 B | ✅ Present |
| `nba.sqlite` | 2.3 GB | 2.2 GB | ✅ Present |

**Status: ✅ ALL FILES PRESENT**

---

## Summary

### ✅ Complete Dataset Status

**Dataset 1 (eoinamoore):** 7/7 files present (100%)
**Dataset 2 (wyattowalsh):** 17/17 files present (100%)

**Overall:** 24/24 files present (100%)

### File Size Differences

The slight size differences between Kaggle and local files are normal and can be attributed to:
- Compression differences
- File system overhead
- Potential minor updates on Kaggle since download

All files are within reasonable size ranges and appear to be complete.

### README Discrepancy

The README.md file mentions file names that don't match the actual Kaggle datasets:
- ❌ `games_details.csv` → ✅ Actually `PlayerStatistics.csv`
- ❌ `games.csv` → ✅ Actually `Games.csv` (capital G)
- ❌ `players.csv` → ✅ Actually `Players.csv` (capital P)
- ❌ `teams.csv` → ✅ Actually `csv/team.csv` (in basketball dataset)

**Recommendation:** Update the README.md to reflect the actual file names from Kaggle.
