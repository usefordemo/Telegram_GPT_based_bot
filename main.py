import asyncio, nest_asyncio
from tg_bot import main

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())