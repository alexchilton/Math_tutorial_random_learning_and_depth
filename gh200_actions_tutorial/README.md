# GH-200 GitHub Actions Tutorial — Testable Workflows

This repository contains **10 testable workflow files** and a comprehensive **Jupyter notebook**
covering all 85 questions from the GH-200 (GitHub Actions) certification practice exam.

## 🚀 Quick Start

1. Create a new GitHub repository
2. Copy everything from this directory into your repo root
3. Push to GitHub
4. Go to the **Actions** tab to see workflows

```bash
# Clone your new repo, then copy these files in
cp -r .github/ /path/to/your-repo/
cd /path/to/your-repo
git add .github/
git commit -m "Add GH-200 tutorial workflows"
git push
```

## 📁 Workflow Files

| # | File | Concepts Covered | Questions |
|---|---|---|---|
| 01 | `triggers-and-events.yml` | Push, PR, schedule, manual dispatch, repository_dispatch | Q7, Q9, Q14, Q19, Q20, Q24, Q31, Q43, Q51 |
| 02 | `env-vars-and-outputs.yml` | Env var scopes, GITHUB_ENV, GITHUB_OUTPUT, job outputs, GITHUB_TOKEN | Q3, Q10, Q13, Q15, Q16, Q18, Q23, Q26, Q27, Q28, Q30, Q35, Q37, Q42, Q49 |
| 03 | `matrix-and-conditions.yml` | Matrix strategy, conditional jobs/steps, multi-line commands | Q11, Q17, Q44 |
| 04 | `service-containers.yml` | Redis service container (ephemeral) | Q5 |
| 05 | `caching-and-artifacts.yml` | Dependency caching, artifact upload/download, retention | Q33, Q39, Q40, Q47, Q48, Q59 |
| 06 | `concurrency-and-environments.yml` | Concurrency groups, environment protection, approvals | Q34, Q36, Q38, Q50 |
| 07 | `reusable-workflow.yml` + `caller-workflow.yml` | Reusable workflows, passing inputs/secrets | Q46, Q77 |
| 08 | `workflow-commands.yml` | Debug, warning, error, masking, grouping, outputs | Q1, Q2, Q41, Q42, Q55, Q72 |
| 09 | `docker-build-push.yml` | Build and push to GitHub Container Registry | Q71 |
| 10 | `workflow-run-chain.yml` | Trigger after another workflow completes | Q14 |

## 🧪 Testing Each Workflow

### Manual Trigger Workflows
Click **Actions** → select the workflow → **Run workflow**:
- 01 - Triggers & Events Demo
- 02 - Env Vars & Outputs Demo
- All others with `workflow_dispatch`

### Push-Triggered Workflows
Make a commit to `main` branch to trigger push-based workflows.

### Environment Approvals (Workflow 06)
1. Go to repo **Settings** → **Environments**
2. Create environment named `staging`
3. Add yourself as a **Required reviewer**
4. Push a commit — the deploy job will wait for your approval

## 📓 Jupyter Notebook

The `GH200_GitHub_Actions_Tutorial.ipynb` notebook contains:
- All 85 exam questions organized by topic
- Detailed explanations for every answer
- Working YAML examples
- Complete answer key
- Exam tips and cheat sheets

## 📋 Topic Coverage

| Topic | Weight | Questions |
|---|---|---|
| Workflow Triggers & Events | ~15% | Q7, Q9, Q14, Q19, Q20, Q24, Q31, Q43, Q51 |
| Env Vars, Secrets & Outputs | ~25% | Q3, Q10, Q13, Q15–Q16, Q18, Q21, Q23, Q26–Q28, Q30, Q35, Q37, Q42, Q49, Q62, Q75, Q77, Q80 |
| Runners | ~10% | Q4–Q6, Q22, Q69, Q70, Q78, Q82 |
| Workflow Commands & Debug | ~6% | Q1, Q2, Q41, Q55, Q72 |
| Conditionals, Matrix & Concurrency | ~12% | Q11, Q17, Q34, Q36, Q38, Q44, Q45, Q47–Q50 |
| Custom Actions Development | ~25% | Q8, Q12, Q25, Q46, Q52–Q58, Q60–Q61, Q64, Q66–Q68, Q73–Q74, Q76, Q81, Q85 |
| Artifacts & Caching | ~7% | Q32, Q33, Q39, Q40, Q59, Q71 |
| Enterprise & Organization | ~7% | Q29, Q63, Q65, Q79, Q83, Q84 |
