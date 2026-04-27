# Claude Code 接入 DeepSeek 配置指南

## 配置步骤

1. **获取 DeepSeek API 密钥**
   - 访问 [DeepSeek 官网](https://platform.deepseek.com/)
   - 注册账号并获取 API 密钥

2. **配置 VS Code 设置**
   - 打开 VS Code
   - 按 `Ctrl+Shift+P` 打开命令面板
   - 输入 `Preferences: Open Settings (JSON)` 并选择
   - 在 `settings.json` 文件中添加以下配置：

```json
{
    "claude-code.apiUrl": "https://api.deepseek.com/v1",
    "claude-code.apiKey": "你的DeepSeek API密钥",
    "claude-code.model": "deepseek-chat"
}
```

3. **重启 VS Code**
   - 保存设置文件后重启 VS Code 使配置生效

## 使用方法

配置完成后，你就可以在 VS Code 中使用 Claude Code 插件了。它会使用 DeepSeek 的 AI 模型来帮助你编程。

## 注意事项

- API 密钥请妥善保管，不要泄露给他人
- DeepSeek API 有使用限制，请查看官网的计费和限制说明
- 如果遇到连接问题，请检查网络和 API 密钥是否正确