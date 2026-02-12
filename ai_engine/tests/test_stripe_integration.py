"""
QBITEL - Stripe Integration Tests

Tests for Stripe Connect payment processing, revenue sharing, and webhooks.
"""

import pytest
import uuid
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from ai_engine.marketplace.stripe_integration import StripeConnectManager, get_stripe_manager
from ai_engine.models.marketplace import (
    MarketplaceProtocol,
    MarketplaceUser,
    MarketplaceTransaction,
    ProtocolStatus,
    UserType,
)


@pytest.fixture
def mock_stripe_config():
    """Mock Stripe configuration."""
    with patch("ai_engine.marketplace.stripe_integration.get_config") as mock_config:
        config = Mock()
        config.marketplace.stripe_api_key = "sk_test_mock_key"
        config.marketplace.platform_fee = 0.30
        config.marketplace.stripe_webhook_secret = "whsec_test_secret"
        mock_config.return_value = config
        yield config


@pytest.fixture
def stripe_manager(mock_stripe_config):
    """Create StripeConnectManager instance with mocked config."""
    manager = StripeConnectManager()
    return manager


@pytest.fixture
def mock_protocol():
    """Mock marketplace protocol."""
    protocol = Mock(spec=MarketplaceProtocol)
    protocol.protocol_id = uuid.uuid4()
    protocol.protocol_name = "test-protocol"
    protocol.display_name = "Test Protocol"
    protocol.short_description = "A test protocol"
    protocol.base_price = Decimal("99.99")
    protocol.price_model = Mock(value="one_time")
    protocol.author = Mock()
    protocol.author.stripe_account_id = "acct_test_creator"
    protocol.author.username = "test_creator"
    return protocol


@pytest.fixture
def mock_customer():
    """Mock customer user."""
    customer = Mock()
    customer.id = uuid.uuid4()
    customer.email = "customer@example.com"
    return customer


class TestStripeConnectManager:
    """Test Stripe Connect manager functionality."""

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Account.create")
    @patch("ai_engine.marketplace.stripe_integration.stripe.AccountLink.create")
    async def test_create_connected_account(self, mock_account_link, mock_account_create, stripe_manager):
        """Test creating a Stripe Connect account for protocol creator."""
        # Mock Stripe API responses
        mock_account = Mock()
        mock_account.id = "acct_test_123"
        mock_account_create.return_value = mock_account

        mock_link = Mock()
        mock_link.url = "https://connect.stripe.com/onboarding/123"
        mock_account_link.return_value = mock_link

        # Create connected account
        user_id = uuid.uuid4()
        result = await stripe_manager.create_connected_account(
            user_id=user_id, email="creator@example.com", country="US", business_type="individual"
        )

        # Verify result
        assert result["created"] is True
        assert result["account_id"] == "acct_test_123"
        assert result["onboarding_url"] == "https://connect.stripe.com/onboarding/123"

        # Verify Stripe API was called correctly
        mock_account_create.assert_called_once()
        call_kwargs = mock_account_create.call_args[1]
        assert call_kwargs["type"] == "express"
        assert call_kwargs["email"] == "creator@example.com"
        assert call_kwargs["metadata"]["user_id"] == str(user_id)

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.PaymentIntent.create")
    @patch("ai_engine.marketplace.stripe_integration.get_database_manager")
    async def test_process_one_time_payment(
        self, mock_db_manager, mock_payment_intent, stripe_manager, mock_protocol, mock_customer
    ):
        """Test processing a one-time payment with revenue sharing."""
        # Mock database session
        mock_session = Mock()
        mock_db_manager.return_value.get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_protocol

        # Mock Stripe PaymentIntent
        mock_intent = Mock()
        mock_intent.id = "pi_test_123"
        mock_intent.status = "succeeded"
        mock_intent.charges.data = [Mock(id="ch_test_123")]
        mock_payment_intent.return_value = mock_intent

        # Process payment
        result = await stripe_manager.process_one_time_payment(
            protocol_id=mock_protocol.protocol_id,
            customer_id=mock_customer.id,
            payment_method_id="pm_test_card",
            amount=Decimal("99.99"),
            currency="usd",
        )

        # Verify result
        assert result["success"] is True
        assert result["payment_intent_id"] == "pi_test_123"
        assert result["status"] == "succeeded"
        assert result["amount"] == Decimal("99.99")

        # Verify revenue split: 30% platform, 70% creator
        assert result["platform_fee"] == 29.997  # 30% of $99.99
        assert result["creator_revenue"] == 69.993  # 70% of $99.99

        # Verify Stripe was called with correct fees
        mock_payment_intent.assert_called_once()
        call_kwargs = mock_payment_intent.call_args[1]
        assert call_kwargs["amount"] == 9999  # $99.99 in cents
        assert call_kwargs["application_fee_amount"] == 2999  # 30% fee in cents
        assert call_kwargs["transfer_data"]["destination"] == "acct_test_creator"

        # Verify transaction was saved
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Subscription.create")
    @patch("ai_engine.marketplace.stripe_integration.stripe.Customer.create")
    @patch("ai_engine.marketplace.stripe_integration.get_database_manager")
    async def test_create_subscription(
        self, mock_db_manager, mock_customer_create, mock_subscription_create, stripe_manager, mock_protocol, mock_customer
    ):
        """Test creating a subscription with revenue sharing."""
        # Mock database session
        mock_session = Mock()
        mock_db_manager.return_value.get_session.return_value = mock_session

        # Create a mock user with email attribute
        mock_user = Mock()
        mock_user.email = "customer@example.com"
        mock_user.stripe_customer_id = None

        # Mock database queries to return different values based on what's being queried
        def mock_query(model):
            query_mock = Mock()
            filter_mock = Mock()

            # Return protocol for MarketplaceProtocol queries
            if hasattr(model, "__name__") and "Protocol" in model.__name__:
                filter_mock.first.return_value = mock_protocol
            # Return user for User queries
            else:
                filter_mock.first.return_value = mock_user

            query_mock.filter.return_value = filter_mock
            return query_mock

        mock_session.query = mock_query

        # Mock Stripe Customer creation
        stripe_customer = Mock()
        stripe_customer.id = "cus_test_123"
        mock_customer_create.return_value = stripe_customer

        # Mock Stripe Subscription
        mock_sub = Mock()
        mock_sub.id = "sub_test_123"
        mock_sub.status = "active"
        mock_sub.current_period_end = 1234567890
        mock_subscription_create.return_value = mock_sub

        # Create subscription
        result = await stripe_manager.create_subscription(
            protocol_id=mock_protocol.protocol_id,
            customer_id=mock_customer.id,
            payment_method_id="pm_test_card",
            price_id="price_test_123",
        )

        # Verify result
        assert result["success"] is True
        assert result["subscription_id"] == "sub_test_123"
        assert result["status"] == "active"
        assert result["current_period_end"] == 1234567890

        # Verify Stripe subscription was created with platform fee
        mock_subscription_create.assert_called_once()
        call_kwargs = mock_subscription_create.call_args[1]
        assert call_kwargs["application_fee_percent"] == 30  # 30% platform fee
        assert call_kwargs["transfer_data"]["destination"] == "acct_test_creator"

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Subscription.modify")
    async def test_cancel_subscription_at_period_end(self, mock_subscription_modify, stripe_manager):
        """Test cancelling subscription at end of billing period."""
        # Mock Stripe Subscription
        mock_sub = Mock()
        mock_sub.id = "sub_test_123"
        mock_sub.status = "active"
        mock_sub.cancel_at_period_end = True
        mock_subscription_modify.return_value = mock_sub

        # Cancel subscription
        result = await stripe_manager.cancel_subscription(subscription_id="sub_test_123", at_period_end=True)

        # Verify result
        assert result["success"] is True
        assert result["subscription_id"] == "sub_test_123"
        assert result["cancel_at_period_end"] is True

        # Verify Stripe was called correctly
        mock_subscription_modify.assert_called_once_with("sub_test_123", cancel_at_period_end=True)

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Subscription.delete")
    async def test_cancel_subscription_immediately(self, mock_subscription_delete, stripe_manager):
        """Test cancelling subscription immediately."""
        # Mock Stripe Subscription
        mock_sub = Mock()
        mock_sub.id = "sub_test_123"
        mock_sub.status = "canceled"
        mock_sub.cancel_at_period_end = False
        mock_subscription_delete.return_value = mock_sub

        # Cancel subscription immediately
        result = await stripe_manager.cancel_subscription(subscription_id="sub_test_123", at_period_end=False)

        # Verify result
        assert result["success"] is True
        assert result["status"] == "canceled"

        # Verify Stripe was called correctly
        mock_subscription_delete.assert_called_once_with("sub_test_123")

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Webhook.construct_event")
    async def test_handle_webhook_payment_succeeded(self, mock_construct_event, stripe_manager):
        """Test handling successful payment webhook."""
        # Mock webhook event
        event = {
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test_123",
                    "metadata": {"protocol_id": str(uuid.uuid4()), "customer_id": str(uuid.uuid4())},
                }
            },
        }
        mock_construct_event.return_value = event

        # Handle webhook
        result = await stripe_manager.handle_webhook(payload=b"webhook_payload", signature="test_signature")

        # Verify result
        assert result["handled"] is True
        assert result["event"] == "payment_succeeded"

        # Verify webhook was validated
        mock_construct_event.assert_called_once()

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.Webhook.construct_event")
    async def test_handle_webhook_subscription_created(self, mock_construct_event, stripe_manager):
        """Test handling subscription created webhook."""
        # Mock webhook event
        event = {"type": "customer.subscription.created", "data": {"object": {"id": "sub_test_123", "status": "active"}}}
        mock_construct_event.return_value = event

        # Handle webhook
        result = await stripe_manager.handle_webhook(payload=b"webhook_payload", signature="test_signature")

        # Verify result
        assert result["handled"] is True
        assert result["event"] == "subscription_created"

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.stripe.PaymentIntent.create")
    @patch("ai_engine.marketplace.stripe_integration.get_database_manager")
    async def test_payment_failure(self, mock_db_manager, mock_payment_intent, stripe_manager, mock_protocol, mock_customer):
        """Test handling payment failure."""
        # Mock database session
        mock_session = Mock()
        mock_db_manager.return_value.get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_protocol

        # Mock Stripe error
        mock_payment_intent.side_effect = Exception("Card declined")

        # Process payment should raise exception
        with pytest.raises(Exception, match="Card declined"):
            await stripe_manager.process_one_time_payment(
                protocol_id=mock_protocol.protocol_id,
                customer_id=mock_customer.id,
                payment_method_id="pm_test_card",
                amount=Decimal("99.99"),
                currency="usd",
            )

        # Verify session was rolled back
        mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.get_database_manager")
    async def test_protocol_not_found(self, mock_db_manager, stripe_manager, mock_customer):
        """Test payment processing when protocol doesn't exist."""
        # Mock database session returning None
        mock_session = Mock()
        mock_db_manager.return_value.get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Process payment should raise ValueError
        with pytest.raises(ValueError, match="Protocol .* not found"):
            await stripe_manager.process_one_time_payment(
                protocol_id=uuid.uuid4(),
                customer_id=mock_customer.id,
                payment_method_id="pm_test_card",
                amount=Decimal("99.99"),
                currency="usd",
            )

    @pytest.mark.asyncio
    @patch("ai_engine.marketplace.stripe_integration.get_database_manager")
    async def test_creator_without_stripe_account(self, mock_db_manager, stripe_manager, mock_protocol, mock_customer):
        """Test payment processing when creator has no Stripe account."""
        # Mock protocol with creator that has no Stripe account
        mock_protocol.author.stripe_account_id = None

        mock_session = Mock()
        mock_db_manager.return_value.get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_protocol

        # Process payment should raise ValueError
        with pytest.raises(ValueError, match="has no Stripe account"):
            await stripe_manager.process_one_time_payment(
                protocol_id=mock_protocol.protocol_id,
                customer_id=mock_customer.id,
                payment_method_id="pm_test_card",
                amount=Decimal("99.99"),
                currency="usd",
            )


class TestStripeSingleton:
    """Test Stripe manager singleton pattern."""

    def test_get_stripe_manager_singleton(self):
        """Test that get_stripe_manager returns the same instance."""
        # Clear the singleton
        import ai_engine.marketplace.stripe_integration as stripe_module

        stripe_module._stripe_manager = None

        with patch("ai_engine.marketplace.stripe_integration.get_config") as mock_config:
            config = Mock()
            config.marketplace.stripe_api_key = "sk_test_mock"
            config.marketplace.platform_fee = 0.30
            mock_config.return_value = config

            # Get manager twice
            manager1 = get_stripe_manager()
            manager2 = get_stripe_manager()

            # Should be the same instance
            assert manager1 is manager2


class TestRevenueCalculations:
    """Test revenue split calculations."""

    @pytest.mark.parametrize(
        "amount,platform_fee,expected_platform,expected_creator",
        [
            (Decimal("100.00"), 0.30, 30.00, 70.00),
            (Decimal("99.99"), 0.30, 29.997, 69.993),
            (Decimal("49.99"), 0.30, 14.997, 34.993),
            (Decimal("9.99"), 0.30, 2.997, 6.993),
            (Decimal("0.99"), 0.30, 0.297, 0.693),
            (Decimal("100.00"), 0.20, 20.00, 80.00),  # Different fee rate
            (Decimal("100.00"), 0.15, 15.00, 85.00),  # Lower fee rate
        ],
    )
    def test_revenue_split_calculations(self, amount, platform_fee, expected_platform, expected_creator):
        """Test revenue split calculations for different amounts and fee rates."""
        # Convert to cents
        amount_cents = int(amount * 100)
        platform_fee_cents = int(amount_cents * platform_fee)
        creator_amount_cents = amount_cents - platform_fee_cents

        # Convert back to dollars
        platform_revenue = platform_fee_cents / 100
        creator_revenue = creator_amount_cents / 100

        # Verify calculations (allow 1 cent tolerance due to rounding)
        assert abs(platform_revenue - expected_platform) < 0.01
        assert abs(creator_revenue - expected_creator) < 0.01
        assert abs((platform_revenue + creator_revenue) - float(amount)) < 0.01
