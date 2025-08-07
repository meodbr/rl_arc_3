import aiohttp
import numpy as np

class AsyncAPIEnv:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = aiohttp.ClientSession()

    async def reset(self):
        async with self.session.post(f"{self.api_url}/reset") as resp:
            data = await resp.json()
            return np.array(data["observation"], dtype=np.float32)

    async def step(self, action):
        async with self.session.post(f"{self.api_url}/step", json={"action": int(action)}) as resp:
            data = await resp.json()
            obs = np.array(data["observation"], dtype=np.float32)
            reward = data["reward"]
            done = data["done"]
            info = data.get("info", {})
            return obs, reward, done, info

    async def close(self):
        await self.session.close()