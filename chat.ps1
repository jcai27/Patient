# Chat with VirtualHuman persona
param(
    [string]$Message = "How are you doing?"
)

Write-Host "💬 Chatting with VirtualHuman..." -ForegroundColor Cyan
Write-Host ""

$body = @{
    user_id = "user1"
    message = $Message
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "http://localhost:8000/chat" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body
    
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host $result.response -ForegroundColor White
    Write-Host ""
    if ($result.citations.Count -gt 0) {
        Write-Host "📚 Citations: $($result.citations -join ', ')" -ForegroundColor Cyan
    }
    if ($result.scores) {
        Write-Host "📊 Scores: Fact=$($result.scores.factuality), Persona=$($result.scores.persona), Help=$($result.scores.helpfulness), Safe=$($result.scores.safety)" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "💡 To chat again, run:" -ForegroundColor Gray
    Write-Host "   .\chat.ps1 -Message 'Your message here'" -ForegroundColor Gray
} catch {
    Write-Host ""
    Write-Host "❌ Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails) {
        Write-Host $_.ErrorDetails.Message -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "💡 Make sure:" -ForegroundColor Yellow
    Write-Host "   1. Server is running (python run_server.py)" -ForegroundColor Yellow
    Write-Host "   2. Transcript is ingested (.\ingest_now.ps1)" -ForegroundColor Yellow
}

