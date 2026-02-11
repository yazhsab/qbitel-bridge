#!/usr/bin/env python3
"""
QBITEL - Stripe Webhook Configuration Script

Automatically configures Stripe webhooks for the marketplace.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import ai_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import stripe
except ImportError:
    print("ERROR: stripe not installed. Run: pip3 install stripe")
    sys.exit(1)

from ai_engine.core.config import get_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_webhook_endpoint(url: str, api_key: str) -> dict:
    """
    Create a webhook endpoint in Stripe.

    Args:
        url: The webhook endpoint URL
        api_key: Stripe API key

    Returns:
        Dict with webhook details
    """
    try:
        stripe.api_key = api_key

        logger.info(f"Creating webhook endpoint: {url}")

        # Events to listen for
        events = [
            # Payment events
            'payment_intent.succeeded',
            'payment_intent.payment_failed',
            'payment_intent.canceled',
            'charge.succeeded',
            'charge.failed',
            'charge.refunded',

            # Subscription events
            'customer.subscription.created',
            'customer.subscription.updated',
            'customer.subscription.deleted',
            'customer.subscription.trial_will_end',
            'invoice.payment_succeeded',
            'invoice.payment_failed',

            # Connect account events
            'account.updated',
            'account.application.authorized',
            'account.application.deauthorized',

            # Payout events
            'payout.paid',
            'payout.failed',
        ]

        webhook = stripe.WebhookEndpoint.create(
            url=url,
            enabled_events=events,
            api_version='2023-10-16'  # Latest stable version
        )

        logger.info(f"✅ Webhook created successfully!")
        logger.info(f"   ID: {webhook.id}")
        logger.info(f"   Secret: {webhook.secret}")

        return {
            'id': webhook.id,
            'secret': webhook.secret,
            'url': webhook.url,
            'status': webhook.status
        }

    except stripe.error.StripeError as e:
        logger.error(f"Failed to create webhook: {e}")
        return None


def list_existing_webhooks(api_key: str) -> list:
    """
    List existing webhook endpoints.

    Args:
        api_key: Stripe API key

    Returns:
        List of webhook endpoints
    """
    try:
        stripe.api_key = api_key

        webhooks = stripe.WebhookEndpoint.list(limit=100)

        return webhooks.data

    except stripe.error.StripeError as e:
        logger.error(f"Failed to list webhooks: {e}")
        return []


def delete_webhook(webhook_id: str, api_key: str) -> bool:
    """
    Delete a webhook endpoint.

    Args:
        webhook_id: Webhook endpoint ID
        api_key: Stripe API key

    Returns:
        True if successful
    """
    try:
        stripe.api_key = api_key

        stripe.WebhookEndpoint.delete(webhook_id)

        logger.info(f"✅ Webhook {webhook_id} deleted")
        return True

    except stripe.error.StripeError as e:
        logger.error(f"Failed to delete webhook: {e}")
        return False


def test_webhook_connection(webhook_url: str) -> bool:
    """
    Test webhook endpoint connectivity.

    Args:
        webhook_url: The webhook URL

    Returns:
        True if reachable
    """
    try:
        import requests

        logger.info("Testing webhook endpoint connectivity...")

        # Make a HEAD request to check if endpoint is reachable
        response = requests.head(webhook_url, timeout=10)

        if response.status_code in [200, 405]:  # 405 is OK (POST expected, not HEAD)
            logger.info("✅ Webhook endpoint is reachable")
            return True
        else:
            logger.warning(f"⚠️  Webhook endpoint returned status {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"⚠️  Could not reach webhook endpoint: {e}")
        logger.info("   This is normal if the server is not running yet")
        return False


def main():
    """Main setup function."""
    print("=" * 70)
    print("QBITEL - Stripe Webhook Configuration")
    print("=" * 70)
    print()

    # Load configuration
    try:
        config = get_config()
        api_key = config.marketplace.stripe_api_key

        if not api_key:
            logger.error("STRIPE_API_KEY not configured!")
            logger.info("\nPlease set the STRIPE_API_KEY environment variable:")
            logger.info("  export STRIPE_API_KEY=sk_test_xxxxx")
            return False

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # Get webhook URL from user
    print("Enter your webhook endpoint URL:")
    print("  Example: https://api.qbitel.com/api/v1/marketplace/webhooks/stripe")
    print("  Example: https://your-domain.com/api/v1/marketplace/webhooks/stripe")
    print()

    webhook_url = input("Webhook URL: ").strip()

    if not webhook_url:
        logger.error("Webhook URL is required!")
        return False

    if not webhook_url.startswith('https://'):
        logger.error("Webhook URL must use HTTPS!")
        logger.info("Stripe requires HTTPS for webhook endpoints")
        return False

    print()

    # Test endpoint connectivity
    test_webhook_connection(webhook_url)
    print()

    # List existing webhooks
    logger.info("Checking for existing webhooks...")
    existing_webhooks = list_existing_webhooks(api_key)

    if existing_webhooks:
        print(f"\nFound {len(existing_webhooks)} existing webhook(s):")
        for wh in existing_webhooks:
            print(f"  - {wh.id}: {wh.url} ({wh.status})")

        print("\nDo you want to:")
        print("  1. Create a new webhook")
        print("  2. Delete existing webhooks and create new")
        print("  3. Exit")

        choice = input("\nChoice (1-3): ").strip()

        if choice == '2':
            for wh in existing_webhooks:
                if input(f"\nDelete {wh.url}? (y/n): ").lower() == 'y':
                    delete_webhook(wh.id, api_key)
        elif choice == '3':
            return True

    print()

    # Create webhook
    result = create_webhook_endpoint(webhook_url, api_key)

    if not result:
        logger.error("❌ Failed to create webhook")
        return False

    print()
    print("=" * 70)
    print("✅ Webhook configuration complete!")
    print("=" * 70)
    print()
    print("IMPORTANT: Save this webhook secret!")
    print()
    print(f"  Webhook ID: {result['id']}")
    print(f"  Webhook Secret: {result['secret']}")
    print()
    print("Add this to your environment:")
    print(f"  export STRIPE_WEBHOOK_SECRET={result['secret']}")
    print()
    print("Or add to .env file:")
    print(f"  STRIPE_WEBHOOK_SECRET={result['secret']}")
    print()
    print("Next steps:")
    print("1. Add STRIPE_WEBHOOK_SECRET to your environment variables")
    print("2. Restart your QBITEL server")
    print("3. Test webhook with: stripe trigger payment_intent.succeeded")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
