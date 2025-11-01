# Ingest VirtualHuman transcript
Write-Host "📤 Starting transcript ingestion..." -ForegroundColor Cyan
Write-Host "   This will take 2-5 minutes (multiple LLM API calls)" -ForegroundColor Yellow
Write-Host ""

$body = @{
    transcript_path = "transcript_cleaned.txt"
    persona_name = "VirtualHuman"
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "http://localhost:8000/ingest/transcript" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body
    
    Write-Host ""
    Write-Host "✅ Ingestion Complete!" -ForegroundColor Green
    Write-Host "   Persona: $($result.persona_name)"
    Write-Host "   Facts extracted: $($result.facts_count)"
    Write-Host "   Examples generated: $($result.examples_count)"
    Write-Host ""
    Write-Host "📁 Persona files created in: persona/VirtualHuman/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Ready to chat! Run: .\chat.ps1" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ Error occurred:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails) {
        Write-Host $_.ErrorDetails.Message -ForegroundColor Red
    }
}

