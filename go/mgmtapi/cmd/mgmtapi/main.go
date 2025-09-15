package main

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/qslb/mgmtapi/internal/devices"
)

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize device lifecycle manager
	lifecycleConfig := devices.DefaultLifecycleConfig()
	dlm, err := devices.NewDeviceLifecycleManager(logger, lifecycleConfig)
	if err != nil {
		logger.Fatal("failed to initialize device lifecycle manager", zap.Error(err))
	}

	r := gin.Default()

	// Device enrollment endpoints
	r.POST("/devices/enroll/start", func(c *gin.Context) {
		var req devices.EnrollmentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		session, err := dlm.StartEnrollment(c.Request.Context(), &req)
		if err != nil {
			logger.Error("enrollment start failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"session_id": session.ID,
			"challenge":  session.Challenge,
			"expires_at": session.ExpiresAt,
		})
	})

	r.POST("/devices/enroll/:session_id/attest", func(c *gin.Context) {
		sessionID := c.Param("session_id")
		
		var attestation devices.AttestationData
		if err := c.ShouldBindJSON(&attestation); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := dlm.SubmitAttestation(c.Request.Context(), sessionID, &attestation)
		if err != nil {
			logger.Error("attestation submission failed", zap.String("session_id", sessionID), zap.Error(err))
			c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "attestation_verified"})
	})

	r.POST("/devices/enroll/:session_id/complete", func(c *gin.Context) {
		sessionID := c.Param("session_id")
		
		// Parse CSR from request body
		var csrData struct {
			CSR string `json:"csr"`
		}
		if err := c.ShouldBindJSON(&csrData); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Parse the CSR (simplified - would need proper PEM parsing)
		csr := &x509.CertificateRequest{} // Placeholder

		cert, err := dlm.CompleteEnrollment(c.Request.Context(), sessionID, csr)
		if err != nil {
			logger.Error("enrollment completion failed", zap.String("session_id", sessionID), zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"certificate_id": cert.ID,
			"certificate":    cert.Certificate,
			"not_after":      cert.NotAfter,
		})
	})

	// Device management endpoints
	r.GET("/devices/:device_id", func(c *gin.Context) {
		deviceID := c.Param("device_id")
		
		device, err := dlm.GetDevice(deviceID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, device)
	})

	r.GET("/organizations/:org_id/devices", func(c *gin.Context) {
		orgID := c.Param("org_id")
		
		devices, err := dlm.ListDevices(orgID, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"devices": devices})
	})

	r.PUT("/devices/:device_id/configuration", func(c *gin.Context) {
		deviceID := c.Param("device_id")
		
		var config map[string]interface{}
		if err := c.ShouldBindJSON(&config); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := dlm.UpdateDeviceConfiguration(c.Request.Context(), deviceID, config)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "configuration_updated"})
	})

	r.POST("/devices/:device_id/suspend", func(c *gin.Context) {
		deviceID := c.Param("device_id")
		
		var req struct {
			Reason string `json:"reason"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := dlm.SuspendDevice(c.Request.Context(), deviceID, req.Reason)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "device_suspended"})
	})

	r.POST("/devices/:device_id/reactivate", func(c *gin.Context) {
		deviceID := c.Param("device_id")
		
		err := dlm.ReactivateDevice(c.Request.Context(), deviceID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "device_reactivated"})
	})

	r.POST("/devices/:device_id/decommission", func(c *gin.Context) {
		deviceID := c.Param("device_id")
		
		var req struct {
			Reason string `json:"reason"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := dlm.DecommissionDevice(c.Request.Context(), deviceID, req.Reason)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "device_decommissioned"})
	})

	// OTA update endpoints
	r.POST("/ota/push", func(c *gin.Context) {
		var req struct {
			DeviceID    string                 `json:"device_id"`
			UpdateType  string                 `json:"update_type"`
			Version     string                 `json:"version"`
			PackageURL  string                 `json:"package_url"`
			Checksum    string                 `json:"checksum"`
			Metadata    map[string]interface{} `json:"metadata"`
		}
		
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// TODO: Implement OTA update scheduling
		logger.Info("OTA update scheduled",
			zap.String("device_id", req.DeviceID),
			zap.String("version", req.Version),
			zap.String("update_type", req.UpdateType))

		c.JSON(http.StatusOK, gin.H{
			"scheduled": true,
			"update_id": "ota-" + req.DeviceID + "-" + req.Version,
		})
	})

	// Health and status endpoints
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"timestamp": time.Now(),
			"version":   "1.0.0",
		})
	})

	r.GET("/status", func(c *gin.Context) {
		// Get system status
		c.JSON(http.StatusOK, gin.H{
			"service":   "mgmt-api",
			"status":    "running",
			"timestamp": time.Now(),
			"uptime":    time.Since(time.Now()), // Would track actual uptime
		})
	})

	addr := getenv("MGMT_ADDR", ":8081")
	logger.Info("mgmt api listening",
		zap.String("addr", addr),
		zap.Bool("tpm_attestation_required", lifecycleConfig.RequireTPMAttestation),
		zap.Duration("certificate_validity", lifecycleConfig.CertificateValidity))
	
	r.Run(addr)
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" { return v }
	return def
}
