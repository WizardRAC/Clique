"""Mock WandB client for testing without Lambda service access."""
import time
from bittensor import Wallet
from common.base.wandb_logging.model import (
    WandbRunInitResponse,
    WandbRunLogData,
    WandbRunLogResponse,
)


class MockWandbClient:
    """Mock WandB client that logs locally instead of to Lambda service."""
    
    def __init__(self, wallet: Wallet, base_version: str, netuid: int):
        self.wallet = wallet
        self.netuid = netuid
        self.hotkey = wallet.hotkey.ss58_address
        self.base_version = str(base_version)
        self.run_id = None
        self.validator_version = None
        print(f"[MockWandbClient] Initialized for hotkey {self.hotkey} on netuid {netuid}")

    def init(self, version: str) -> WandbRunInitResponse:
        """Mock initialization - generates local run ID."""
        self.validator_version = str(version)
        self.run_id = f"mock_run_{int(time.time())}"
        print(f"[MockWandbClient] Initialized run {self.run_id} with version {version}")
        return WandbRunInitResponse(run_id=self.run_id)

    def log(self, data: WandbRunLogData) -> WandbRunLogResponse:
        """Mock logging - prints to console instead of sending to Lambda."""
        if not self.run_id:
            raise ValueError("WandB run is not started.")
        
        print(f"[MockWandbClient] Logging data for run {self.run_id}:")
        print(f"  - UUID: {data.uuid}")
        print(f"  - Type: {data.type}")
        print(f"  - Label: {data.label}")
        print(f"  - Difficulty: {data.difficulty}")
        print(f"  - Nodes: {data.number_of_nodes}")
        print(f"  - Selected miners: {len(data.miner_uids)}")
        print(f"  - Rewards: {data.miner_rewards}")
        
        return WandbRunLogResponse(success=True)

    def finish(self) -> WandbRunInitResponse:
        """Mock finish - cleans up local run."""
        if not self.run_id:
            raise ValueError("WandB run is not started.")
        
        print(f"[MockWandbClient] Finished run {self.run_id}")
        run_id = self.run_id
        self.run_id = None
        return WandbRunInitResponse(run_id=run_id)
