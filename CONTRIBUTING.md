以下是根据您的要求，在原有基础上大幅扩展和细化的**中英双语、完全匿名、高度结构化、可被任何开源项目直接复制使用的贡献指南（Contributing Guide）模板**。

本模板旨在为贡献者提供无歧义、可逐步跟随的操作指南。

```markdown
# 贡献指南 | Contributing Guide

## 欢迎与概述 | Welcome & Overview
感谢你考虑为这个项目做出贡献！无论你是想修复错别字、改进文档、报告问题，还是添加新功能，你的参与都至关重要。
Thank you for considering contributing to this project! Whether you want to fix a typo, improve documentation, report an issue, or add a new feature, your participation is vital.

为确保贡献过程顺畅高效，请仔细阅读本指南。它将引导你完成从发现问题到代码合并的完整流程。
To ensure a smooth and efficient contribution process, please read this guide carefully. It will walk you through the complete workflow from identifying an issue to getting your code merged.

## 开始之前 | Before You Start
1.  **沟通**：在开始进行重大的编码工作之前，请先查看项目的 **议题（Issues）** 列表，确认是否已有相关讨论。如果没有，建议**先创建一个新议题来描述你的想法或问题**，与维护者讨论可行性。这可以避免你的工作与项目方向不符。
    **Communication**: Before starting significant coding work, check the project's **Issues** list to see if there‘s already a related discussion. If not, it‘s recommended to **first create a new issue** to describe your idea or problem and discuss its feasibility with the maintainers. This can prevent your work from diverging from the project‘s direction.
2.  **熟悉项目**：请花一些时间阅读项目的 `README.md` 文件，了解项目的目标、结构和代码风格。
    **Familiarize Yourself with the Project**: Take some time to read the project‘s `README.md` file to understand its goals, structure, and coding style.

## 贡献流程详解 | Detailed Contribution Workflow
请严格按照以下步骤操作，这是使用 Git 和 GitHub 进行贡献的标准流程。
Please follow these steps strictly. This is the standard workflow for contributing using Git and GitHub.

### 步骤 1：复刻仓库 | Step 1: Fork the Repository
1.  访问项目在 GitHub 上的主页。
    Visit the project‘s homepage on GitHub.
2.  点击页面右上角的 **“Fork”** 按钮。这将在你的个人 GitHub 账户下创建一个该项目的独立副本。
    Click the **“Fork”** button in the top-right corner of the page. This will create a standalone copy of the project under your personal GitHub account.

### 步骤 2：克隆到本地 | Step 2: Clone to Your Local Machine
1.  进入你复刻后仓库的页面（位于你的账户下）。
    Go to the page of your forked repository (under your account).
2.  点击绿色的 **“Code”** 按钮，复制仓库的 HTTPS 或 SSH 地址。
    Click the green **“Code”** button and copy the repository‘s HTTPS or SSH URL.
3.  在你的电脑终端（命令行）中，执行以下命令，将仓库下载到本地：
    On your computer‘s terminal (command line), run the following command to download the repository locally:
    ```bash
    git clone <你刚才复制的地址>
    git clone <the-url-you-copied>
    ```
4.  进入下载好的项目目录：
    Navigate into the downloaded project directory:
    ```bash
    cd <项目目录名>
    cd <project-directory-name>
    ```

### 步骤 3：创建功能分支 | Step 3: Create a Feature Branch
**永远不要在主分支（`main` 或 `master`）上直接修改。** 为每个新功能或修复创建一个独立的分支。
**Never make changes directly on the main branch (`main` or `master`).** Create a separate branch for each new feature or fix.
1.  创建一个新分支。建议使用描述性的名称。
    Create a new branch. It‘s recommended to use a descriptive name.
    ```bash
    git checkout -b feat/your-feature-name   # 用于新功能 | For a new feature
    # 或 | or
    git checkout -b fix/description-of-bug   # 用于修复问题 | For a bug fix
    # 或 | or
    git checkout -b docs/improve-readme      # 用于改进文档 | For improving documentation
    ```

### 步骤 4：进行你的修改 | Step 4: Make Your Changes
现在你可以在本地代码编辑器中进行修改了。
Now you can make changes in your local code editor.

### 步骤 5：提交更改 | Step 5: Commit Your Changes
1.  保存你的修改。
    Save your changes.
2.  使用 `git add` 命令将修改的文件添加到暂存区。
    Use the `git add` command to stage the modified files.
    ```bash
    git add 文件名.py  # 添加特定文件 | Add a specific file
    # 或 | or
    git add .          # 添加所有当前更改 | Add all current changes
    ```
3.  使用 `git commit` 命令提交更改，并附上一条清晰简明的提交信息。
    Use the `git commit` command to commit the changes with a clear and concise commit message.
    ```bash
    git commit -m "类型: 用一句话描述更改 (关联议题号 #123)"
    git commit -m “type: Describe the change in one sentence (Ref #123)”
    ```
    **提交信息规范**：开头用关键词（如 `feat:`、`fix:`、`docs:`、`style:`），后接简短描述。若有关联的议题（Issue），可在末尾用 `(#123)` 或 `(Ref #123)` 注明。
    **Commit Message Convention**: Start with a keyword (e.g., `feat:`, `fix:`, `docs:`, `style:`), followed by a brief description. If related to an issue, reference it at the end with `(#123)` or `(Ref #123)`.

### 步骤 6：推送到你的复刻仓库 | Step 6: Push to Your Fork
将你的本地分支推送到 GitHub 上你的复刻仓库中。
Push your local branch to your forked repository on GitHub.
```bash
git push origin 你的分支名
git push origin your-branch-name
```

### 步骤 7：发起拉取请求 | Step 7: Open a Pull Request
1.  访问你复刻仓库的 GitHub 页面。
    Visit the GitHub page of your forked repository.
2.  你应该会看到一个按钮，提示你为你刚推送的分支 **“Compare & pull request”**。点击它。
    You should see a button prompting you to **“Compare & pull request”** for the branch you just pushed. Click it.
3.  **仔细填写拉取请求（PR）表单：**
    **Carefully fill out the Pull Request (PR) form:**
    *   **标题**：清晰总结此次 PR 的目的。
        **Title**: Clearly summarize the purpose of this PR.
    *   **描述**：**请使用我们提供的模板**（见下文【PR描述模板】部分），详细说明你做了什么、为什么做，以及如何测试。
        **Description**: **Please use the template we provide** (see the 【PR Description Template】section below) to detail what you did, why, and how it was tested.
4.  确认 PR 是**从你的分支（`你的用户名:你的分支名`）** 合并到**原项目的 `main` 分支**。
    Ensure the PR is merging **from your branch (`your-username:your-branch`)** **into the original project‘s `main` branch**.
5.  点击 **“Create pull request”**。
    Click **“Create pull request”**.

## 贡献内容规范 | Contribution Content Standards
### 对于代码贡献 | For Code Contributions
*   **代码风格**：请遵循项目中已有的代码格式（如缩进、命名）。如果项目有配置文件（如 `.editorconfig`, `.pylintrc`），请遵守。
    **Coding Style**: Follow the existing code formatting in the project (e.g., indentation, naming). Adhere to configuration files (like `.editorconfig`, `.pylintrc`) if present.
*   **注释**：为新代码添加必要的注释，尤其是复杂的逻辑。
    **Comments**: Add necessary comments for new code, especially for complex logic.
*   **可运行性**：确保你提交的代码能够独立运行，没有语法错误。
    **Runnable**: Ensure the code you submit can run independently and has no syntax errors.
*   **测试（如果适用）**：如果项目有测试，请确保你的修改通过了现有测试，并为新功能添加相应的测试。
    **Testing (if applicable)**: If the project has tests, ensure your changes pass the existing tests and add relevant tests for new features.

### 对于文档贡献 | For Documentation Contributions
*   **清晰准确**：语言简洁明了，信息准确无误。
    **Clarity & Accuracy**: Use clear and concise language with accurate information.
*   **格式一致**：遵循项目已有的文档格式（如 Markdown 标题层级、列表样式）。
    **Consistent Formatting**: Follow the project‘s existing documentation format (e.g., Markdown heading hierarchy, list styles).

## PR描述模板 | PR Description Template
在发起 PR 时，请在描述框中复制并填写以下模板：
When opening a PR, please copy and fill out the following template in the description box:

```markdown
## 变更类型 | Type of Change
<!-- 请勾选所有适用的选项。 -->
<!-- Please check all that apply. -->
- [ ] 新功能（非破坏性更新，添加功能）| New feature (non-breaking change which adds functionality)
- [ ] 问题修复（非破坏性更新，修复问题）| Bug fix (non-breaking change which fixes an issue)
- [ ] 文档更新 | Documentation update
- [ ] 代码风格优化（不影响功能）| Code style update (formatting, renaming)
- [ ] 其他（请描述）| Other (please describe):

## 改动的动机与背景 | Motivation and Context
<!-- 为什么需要这个改动？它关联了哪个议题（Issue）？ -->
<!-- Why is this change required? What issue does it relate to? -->
> 关联议题：#XXX | Related Issue: #XXX

## 具体的修改内容 | What Changed
<!-- 请详细描述你具体修改了哪些文件，以及如何实现的。 -->
<!-- Please describe in detail which files you modified and how. -->

## 测试说明 | Testing Performed
<!-- 你做了哪些测试来验证你的修改是有效的？请描述测试步骤。 -->
<!-- What testing did you perform to verify your changes are effective? Describe the steps. -->

## 检查清单 | Checklist
- [ ] 我的代码遵循了项目的代码风格。 | My code follows the project‘s code style.
- [ ] 我已自我检查过我的代码。 | I have reviewed my own code.
- [ ] 我已为我的代码添加了必要的注释。 | I have added necessary comments to my code.
- [ ] 我的修改没有引入新的警告或错误。 | My changes generate no new warnings or errors.
- [ ] 我已将我的修改关联到相应的议题。 | I have linked my changes to the relevant issue.
```

## PR 提交之后 | After Submitting Your PR
*   **代码审查**：项目维护者或其他贡献者可能会在你的 PR 中提出修改建议。请积极参与讨论，并根据反馈进行修改（只需在你本地的分支上继续提交并推送，PR 会自动更新）。
    **Code Review**: Project maintainers or other contributors may suggest changes in your PR. Please participate in the discussion actively and make revisions based on the feedback (just commit and push to your local branch, the PR will update automatically).
*   **合并**：一旦你的 PR 通过审查并获得批准，维护者会将其合并到主分支。恭喜你，你的贡献已成为项目的一部分！
    **Merge**: Once your PR is reviewed and approved, a maintainer will merge it into the main branch. Congratulations, your contribution is now part of the project!

## 获取帮助 | Getting Help
如果在贡献过程中遇到任何问题，你可以：
If you encounter any problems during the contribution process, you can:
1.  在你关联的 **PR 或 Issue 中直接留言评论**。
    Leave a comment directly in the associated **PR or Issue**.
2.  查看项目的其他讨论区（如 GitHub Discussions）。
    Check the project‘s other discussion areas (e.g., GitHub Discussions).

---
感谢你为开源社区付出的时间和精力！
Thank you for your time and effort dedicated to the open-source community!
```