"""
CRONOS AI - Stripe Connect Integration for Marketplace

Handles payment processing, revenue sharing, and subscription management
for the Protocol Marketplace using Stripe Connect.
"""

import logging
import stripe
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from uuid import UUID

from ..core.config import get_config
from ..core.database_manager import get_database_manager
from ..models.marketplace import (
    MarketplaceProtocol,
    MarketplaceUser,
    MarketplaceInstallation,
    MarketplaceTransaction,
    InstallationStatus,
    LicenseType,
    PriceModel,
)

logger = logging.getLogger(__name__)


class StripeConnectManager:
    """
    Manages Stripe Connect operations for marketplace payments.

    Features:
    - Payment processing with revenue sharing
    - Subscription management
    - Webhook handling
    - Connected account management
    """

    def __init__(self):
        """Initialize Stripe Connect manager."""
        self.config = get_config()

        # Initialize Stripe with API key
        stripe.api_key = self.config.marketplace.stripe_api_key

        # Platform fee percentage (e.g., 0.30 for 30%)
        self.platform_fee = self.config.marketplace.platform_fee

        self.logger = logging.getLogger(__name__)

    async def create_connected_account(
        self,
        user_id: UUID,
        email: str,
        country: str = "US",
        business_type: str = "individual"
    ) -> Dict[str, Any]:
        """
        Create a Stripe Connect account for a protocol creator.

        Args:
            user_id: Marketplace user ID
            email: User's email address
            country: Country code (default: US)
            business_type: individual or company

        Returns:
            Dict with account details
        """
        try:
            # Create Express connected account
            account = stripe.Account.create(
                type="express",
                country=country,
                email=email,
                business_type=business_type,
                capabilities={
                    "card_payments": {"requested": True},
                    "transfers": {"requested": True},
                },
                metadata={
                    "user_id": str(user_id),
                    "platform": "cronos-ai-marketplace",
                }
            )

            self.logger.info(f"Created Stripe Connect account for user {user_id}: {account.id}")

            # Create account link for onboarding
            account_link = stripe.AccountLink.create(
                account=account.id,
                refresh_url=f"https://marketplace.cronos-ai.com/onboarding/refresh",
                return_url=f"https://marketplace.cronos-ai.com/onboarding/complete",
                type="account_onboarding",
            )

            return {
                "account_id": account.id,
                "onboarding_url": account_link.url,
                "created": True,
            }

        except Exception as e:
            self.logger.error(f"Stripe Connect account creation failed: {e}")
            raise

    async def process_one_time_payment(
        self,
        protocol_id: UUID,
        customer_id: UUID,
        payment_method_id: str,
        amount: Decimal,
        currency: str = "usd",
    ) -> Dict[str, Any]:
        """
        Process a one-time payment for protocol purchase.

        Args:
            protocol_id: Protocol being purchased
            customer_id: Customer making purchase
            payment_method_id: Stripe payment method ID
            amount: Payment amount
            currency: Currency code (default: usd)

        Returns:
            Dict with payment details and transaction ID
        """
        db_manager = get_database_manager()
        session = db_manager.get_session()

        try:
            # Get protocol and creator info
            protocol = session.query(MarketplaceProtocol).filter(
                MarketplaceProtocol.protocol_id == protocol_id
            ).first()

            if not protocol:
                raise ValueError(f"Protocol {protocol_id} not found")

            # Get creator's Stripe account
            creator = protocol.author
            if not creator.stripe_account_id:
                raise ValueError(f"Creator {creator.username} has no Stripe account")

            # Calculate fees
            total_amount_cents = int(amount * 100)  # Convert to cents
            platform_fee_cents = int(total_amount_cents * self.platform_fee)
            creator_amount_cents = total_amount_cents - platform_fee_cents

            # Create payment intent with automatic transfer
            payment_intent = stripe.PaymentIntent.create(
                amount=total_amount_cents,
                currency=currency,
                payment_method=payment_method_id,
                confirm=True,
                application_fee_amount=platform_fee_cents,
                transfer_data={
                    "destination": creator.stripe_account_id,
                },
                metadata={
                    "protocol_id": str(protocol_id),
                    "customer_id": str(customer_id),
                    "protocol_name": protocol.protocol_name,
                    "transaction_type": "one_time_purchase",
                }
            )

            # Create transaction record
            from uuid import uuid4
            transaction = MarketplaceTransaction(
                transaction_id=uuid4(),
                protocol_id=protocol_id,
                customer_id=customer_id,
                transaction_type="purchase",
                amount=amount,
                currency=currency,
                stripe_payment_intent_id=payment_intent.id,
                stripe_charge_id=payment_intent.charges.data[0].id if payment_intent.charges.data else None,
                platform_fee=Decimal(str(platform_fee_cents / 100)),
                creator_revenue=Decimal(str(creator_amount_cents / 100)),
                status="completed" if payment_intent.status == "succeeded" else "pending",
            )

            session.add(transaction)
            session.commit()

            self.logger.info(
                f"One-time payment processed: {amount} {currency} "
                f"(platform: {platform_fee_cents/100}, creator: {creator_amount_cents/100})"
            )

            return {
                "success": True,
                "transaction_id": transaction.transaction_id,
                "payment_intent_id": payment_intent.id,
                "status": payment_intent.status,
                "amount": amount,
                "platform_fee": platform_fee_cents / 100,
                "creator_revenue": creator_amount_cents / 100,
            }

        except Exception as e:
            self.logger.error(f"Payment processing failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    async def create_subscription(
        self,
        protocol_id: UUID,
        customer_id: UUID,
        payment_method_id: str,
        price_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a subscription for protocol access.

        Args:
            protocol_id: Protocol being subscribed to
            customer_id: Customer subscribing
            payment_method_id: Stripe payment method ID
            price_id: Stripe price ID (created separately)

        Returns:
            Dict with subscription details
        """
        db_manager = get_database_manager()
        session = db_manager.get_session()

        try:
            # Get protocol
            protocol = session.query(MarketplaceProtocol).filter(
                MarketplaceProtocol.protocol_id == protocol_id
            ).first()

            if not protocol:
                raise ValueError(f"Protocol {protocol_id} not found")

            # Get or create Stripe customer
            customer = await self._get_or_create_stripe_customer(customer_id, payment_method_id)

            # Get creator's Stripe account
            creator = protocol.author
            if not creator.stripe_account_id:
                raise ValueError(f"Creator {creator.username} has no Stripe account")

            # Calculate platform fee
            platform_fee_percent = int(self.platform_fee * 100)  # Convert to basis points

            # Create subscription
            subscription = stripe.Subscription.create(
                customer=customer["id"],
                items=[{"price": price_id}] if price_id else [{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": protocol.display_name,
                            "description": protocol.short_description,
                        },
                        "unit_amount": int(protocol.base_price * 100),
                        "recurring": {"interval": "month"},
                    }
                }],
                application_fee_percent=platform_fee_percent,
                transfer_data={
                    "destination": creator.stripe_account_id,
                },
                metadata={
                    "protocol_id": str(protocol_id),
                    "customer_id": str(customer_id),
                    "protocol_name": protocol.protocol_name,
                }
            )

            # Create transaction record
            from uuid import uuid4
            transaction = MarketplaceTransaction(
                transaction_id=uuid4(),
                protocol_id=protocol_id,
                customer_id=customer_id,
                transaction_type="subscription",
                amount=protocol.base_price,
                currency="usd",
                platform_fee=protocol.base_price * Decimal(str(self.platform_fee)),
                creator_revenue=protocol.base_price * Decimal(str(1 - self.platform_fee)),
                status="completed" if subscription.status == "active" else "pending",
            )

            session.add(transaction)
            session.commit()

            self.logger.info(f"Subscription created for protocol {protocol.protocol_name}")

            return {
                "success": True,
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end,
            }

        except Exception as e:
            self.logger.error(f"Subscription creation failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.

        Args:
            subscription_id: Stripe subscription ID
            at_period_end: Cancel at end of billing period (default: True)

        Returns:
            Dict with cancellation details
        """
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)

            self.logger.info(f"Subscription {subscription_id} cancelled")

            return {
                "success": True,
                "subscription_id": subscription_id,
                "status": subscription.status,
                "cancel_at_period_end": subscription.cancel_at_period_end,
            }

        except Exception as e:
            self.logger.error(f"Subscription cancellation failed: {e}")
            raise

    async def handle_webhook(
        self,
        payload: bytes,
        signature: str
    ) -> Dict[str, Any]:
        """
        Handle Stripe webhook events.

        Args:
            payload: Webhook payload
            signature: Stripe signature header

        Returns:
            Dict with processing result
        """
        webhook_secret = self.config.marketplace.stripe_webhook_secret

        try:
            event = stripe.Webhook.construct_event(
                payload, signature, webhook_secret
            )

            # Handle different event types
            event_type = event["type"]

            if event_type == "payment_intent.succeeded":
                return await self._handle_payment_succeeded(event)
            elif event_type == "payment_intent.payment_failed":
                return await self._handle_payment_failed(event)
            elif event_type == "customer.subscription.created":
                return await self._handle_subscription_created(event)
            elif event_type == "customer.subscription.deleted":
                return await self._handle_subscription_deleted(event)
            elif event_type == "invoice.payment_succeeded":
                return await self._handle_invoice_paid(event)
            elif event_type == "account.updated":
                return await self._handle_account_updated(event)
            else:
                self.logger.info(f"Unhandled webhook event: {event_type}")
                return {"handled": False, "event_type": event_type}

        except Exception as e:
            self.logger.error(f"Webhook handling failed: {e}")
            raise

    async def _get_or_create_stripe_customer(
        self,
        customer_id: UUID,
        payment_method_id: str
    ) -> Dict[str, Any]:
        """Get existing Stripe customer or create new one."""
        db_manager = get_database_manager()
        session = db_manager.get_session()

        try:
            # Check if user has Stripe customer ID
            from ..models.database import User
            user = session.query(User).filter(User.id == customer_id).first()

            if user and hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
                return {"id": user.stripe_customer_id}

            # Create new Stripe customer
            customer = stripe.Customer.create(
                email=user.email if user else None,
                payment_method=payment_method_id,
                invoice_settings={
                    "default_payment_method": payment_method_id,
                },
                metadata={
                    "user_id": str(customer_id),
                }
            )

            # Save customer ID
            if user:
                user.stripe_customer_id = customer.id
                session.commit()

            return {"id": customer.id}

        finally:
            session.close()

    async def _handle_payment_succeeded(self, event: Dict) -> Dict[str, Any]:
        """Handle successful payment."""
        payment_intent = event["data"]["object"]
        protocol_id = payment_intent["metadata"].get("protocol_id")

        self.logger.info(f"Payment succeeded for protocol {protocol_id}")

        # Update transaction status in database
        # TODO: Implement transaction status update

        return {"handled": True, "event": "payment_succeeded"}

    async def _handle_payment_failed(self, event: Dict) -> Dict[str, Any]:
        """Handle failed payment."""
        payment_intent = event["data"]["object"]
        protocol_id = payment_intent["metadata"].get("protocol_id")

        self.logger.warning(f"Payment failed for protocol {protocol_id}")

        # Update transaction status and notify user
        # TODO: Implement failure handling

        return {"handled": True, "event": "payment_failed"}

    async def _handle_subscription_created(self, event: Dict) -> Dict[str, Any]:
        """Handle subscription creation."""
        subscription = event["data"]["object"]

        self.logger.info(f"Subscription created: {subscription['id']}")

        return {"handled": True, "event": "subscription_created"}

    async def _handle_subscription_deleted(self, event: Dict) -> Dict[str, Any]:
        """Handle subscription cancellation."""
        subscription = event["data"]["object"]

        self.logger.info(f"Subscription deleted: {subscription['id']}")

        # Update installation status
        # TODO: Implement subscription cleanup

        return {"handled": True, "event": "subscription_deleted"}

    async def _handle_invoice_paid(self, event: Dict) -> Dict[str, Any]:
        """Handle successful invoice payment."""
        invoice = event["data"]["object"]

        self.logger.info(f"Invoice paid: {invoice['id']}")

        return {"handled": True, "event": "invoice_paid"}

    async def _handle_account_updated(self, event: Dict) -> Dict[str, Any]:
        """Handle connected account updates."""
        account = event["data"]["object"]

        self.logger.info(f"Connected account updated: {account['id']}")

        # Update user's payout status if charges_enabled changed
        if account.get("charges_enabled"):
            # Enable payouts for this creator
            pass

        return {"handled": True, "event": "account_updated"}


# Singleton instance
_stripe_manager: Optional[StripeConnectManager] = None


def get_stripe_manager() -> StripeConnectManager:
    """Get singleton Stripe Connect manager instance."""
    global _stripe_manager

    if _stripe_manager is None:
        _stripe_manager = StripeConnectManager()

    return _stripe_manager
