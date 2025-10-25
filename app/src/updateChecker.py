import subprocess
import sys
import os
import logging
from typing import List, Optional, Tuple
from Setting import *

def run_command(cmd, cwd: Optional[str]=None) -> Tuple[int, str, str]:
    """Esegue un comando e ritorna output, codice di uscita."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return 127, "", "Command not found"

def check_git_available() -> bool:
    """Controlla se git è installato."""
    code, out, err = run_command(["git", "--version"])
    if code != 0:
        logger.error("❌ Git non è installato o non è nel PATH.")
        return False
    logger.info(f"✅ Git disponibile: {out}")
    return True

def check_repo(cwd: str) -> bool:
    """Verifica che la directory corrente sia un repo git."""
    code, out, err = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd)
    if code != 0 or out.strip() != "true":
        logger.error("❌ Questa directory non è un repository Git.")
        return False
    return True

def get_current_branch(cwd: str) -> Optional[str]:
    code, out, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return out.strip() if code == 0 else None

def check_updates(cwd, branch="master") -> bool:
    """Controlla se ci sono aggiornamenti remoti."""
    logger.info("🔄 Verifica aggiornamenti da remoto...")
    # aggiorna i riferimenti remoti
    run_command(["git", "fetch", "origin"], cwd)

    # confronto tra locale e remoto
    code1, local_commit, _ = run_command(["git", "rev-parse", branch], cwd)
    code2, remote_commit, _ = run_command(["git", "rev-parse", f"origin/{branch}"], cwd)

    if code1 != 0 or code2 != 0:
        logger.info("⚠️ Impossibile determinare stato dei commit.")
        return False

    if local_commit == remote_commit:
        logger.info("✅ Il codice è già aggiornato.")
        return False
    else:
        logger.info("🟡 Sono disponibili aggiornamenti!")
        return True

def pull_updates(cwd, branch="master") ->bool:
    """Esegue git pull per sincronizzare il codice."""
    logger.info("⬇️ Aggiornamento in corso...")
    code, out, err = run_command(["git", "pull", "origin", branch], cwd)
    
    segments: List[str] = out.replace("\r", '').split("\n")
    for segment in segments:
        logger.info(segment)
    
    if code == 0:
        logger.info("✅ Codice aggiornato con successo.")
        return True
    else:
        logger.info("❌ Errore durante il pull:")
        logger.error(err)
        return False

def auto_update(repo_path=".") -> bool:
    """Funzione principale: verifica e aggiorna il codice."""
    repo_path = os.path.abspath(repo_path)
    logger.info(f"📁 Repository: {repo_path}")

    if not check_git_available():
        return False

    if not check_repo(repo_path):
        return False

    branch = get_current_branch(repo_path)
    if not branch:
        logger.error("⚠️ Impossibile determinare il branch corrente.")
        return False

    if check_updates(repo_path, branch):
        if pull_updates(repo_path, branch):
            return True
    else:
        logger.info("🚀 Procedo con l'esecuzione normale...")
    
    return False


if __name__ == "__main__":
    auto_update(".")
