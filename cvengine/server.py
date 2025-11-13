
import asyncio
import concurrent.futures
import grpc
import logging
import sys

from config import ADDRESS, PORT
from service import CVEngineService
import cvengine_pb2_grpc as cve_grpc

async def serve(shutdown_event: asyncio.Event):
    server = grpc.aio.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        options=[('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    )
    service_servicer = CVEngineService()
    cve_grpc.add_CVEngineServiceServicer_to_server(service_servicer, server)
    server.add_insecure_port(f"{ADDRESS}:{PORT}")
    await server.start()
    logging.info(f"gRPC server started on {ADDRESS}:{PORT}")

    # Wait for shutdown_event to be set
    await shutdown_event.wait()

    # Graceful shutdown
    logging.info("Shutdown requested: stopping gRPC server...")
    await server.stop(grace=5)
    logging.info("Stopping background service components...")
    await service_servicer.shutdown()
    logging.info("Server shutdown complete.")


if __name__ == "__main__":
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()

    # Run the serve coroutine in a task
    serve_task = loop.create_task(serve(shutdown_event))

    try:
        # Run until serve completes (normally it waits on shutdown_event)
        loop.run_until_complete(serve_task)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught in main thread — triggering shutdown event")
        shutdown_event.set()
        # Wait for serve() to finish cleanup
        loop.run_until_complete(serve_task)
    finally:
        loop.close()
